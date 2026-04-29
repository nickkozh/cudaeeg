#include <cuda_runtime.h>
#include "kernels.h"
#include "../taps/fir_taps.h"

namespace eeg {

// Trait that maps Band → constant-memory tap array. Accessing TAPS_<B>[k] from a
// __device__ function gives constant-cache broadcast on warp-uniform k.
template <Band B> struct fir_t;
template <> struct fir_t<BAND_BROADBAND> { static constexpr int N = NTAPS_BROADBAND; __device__ static float at(int i) { return TAPS_BROADBAND[i]; } };
template <> struct fir_t<BAND_DELTA>     { static constexpr int N = NTAPS_DELTA;     __device__ static float at(int i) { return TAPS_DELTA[i];     } };
template <> struct fir_t<BAND_THETA>     { static constexpr int N = NTAPS_THETA;     __device__ static float at(int i) { return TAPS_THETA[i];     } };
template <> struct fir_t<BAND_ALPHA>     { static constexpr int N = NTAPS_ALPHA;     __device__ static float at(int i) { return TAPS_ALPHA[i];     } };
template <> struct fir_t<BAND_BETA>      { static constexpr int N = NTAPS_BETA;      __device__ static float at(int i) { return TAPS_BETA[i];      } };
template <> struct fir_t<BAND_GAMMA>     { static constexpr int N = NTAPS_GAMMA;     __device__ static float at(int i) { return TAPS_GAMMA[i];     } };
template <> struct fir_t<BAND_SSVEP>     { static constexpr int N = NTAPS_SSVEP;     __device__ static float at(int i) { return TAPS_SSVEP[i];     } };

// =============================================================================
// FIR streaming kernel — shared-memory tiling strategy
// =============================================================================
//
// Block assignment: one block per channel. Each block computes the full WIN-sample
// output for its channel and updates the per-channel history state.
//
// Tile contents (single shared-memory tile per block):
//   [0 .. TILE_HISTORY)            — H = N-1 history samples from the previous call
//   [TILE_HISTORY .. TILE_INPUT)   — WIN new input samples for this window
//   [TILE_INPUT  .. TILE_INPUT + TILE_TAPS) — N filter coefficients
//
// Why this size:
//   - WIN matches the pipeline window (a block produces exactly one window's output).
//   - H = N-1 is the minimum history that lets every output in the window be a pure
//     smem dot product — no global-memory re-reads inside the inner loop.
//   - For NTAPS_BROADBAND = 257, smem footprint is (256+256+257)*4 ≈ 3 KB. sm_87 has
//     164 KB smem/SM, so we are at <2% — larger tiles wouldn't buy us anything.
//     Smaller tiles would force history re-reads at tile boundaries.
//
// Memory access pattern:
//   Phase 1 (cooperative load): blockDim.x threads stride-load history+input, then
//                                taps, into smem. Single __syncthreads.
//   Phase 2 (compute): each thread does one length-N dot product over smem. Access
//                      pattern is uniform across threads (no bank conflicts on sm_87).
//   Phase 3 (history save): the last H input samples are written back to global state.
//
// Note on taps: keeping taps in __constant__ memory is *typically* equally fast
// (constant-cache broadcast on warp-uniform k). Staging them into smem is done here
// for explicitness and to give the SM-local scheduler a consistent operand source.
// Performance delta vs constant-cache: noise. Worth measuring with NSight if you care.
// =============================================================================
template <Band B>
__global__ void fir_streaming_kernel(const float* __restrict__ in,
                                     float*       __restrict__ out,
                                     float*       __restrict__ state,
                                     int win)
{
    constexpr int N            = fir_t<B>::N;
    constexpr int TILE_HISTORY = N - 1;
    constexpr int TILE_TAPS    = N;
    extern __shared__ float smem[];   // size: TILE_HISTORY + win + TILE_TAPS

    const int ch  = blockIdx.x;
    const int tid = threadIdx.x;

    const float* in_c    = in    + ch * win;
    float*       out_c   = out   + ch * win;
    float*       state_c = state + ch * TILE_HISTORY;

    float* smem_samples = smem;                             // [TILE_HISTORY + win]
    float* smem_taps    = smem + TILE_HISTORY + win;        // [TILE_TAPS]

    // Phase 1: cooperative load of samples + taps.
    for (int j = tid; j < TILE_HISTORY + win; j += blockDim.x) {
        smem_samples[j] = (j < TILE_HISTORY) ? state_c[j] : in_c[j - TILE_HISTORY];
    }
    for (int k = tid; k < TILE_TAPS; k += blockDim.x) {
        smem_taps[k] = fir_t<B>::at(k);
    }
    __syncthreads();

    // Phase 2: each thread emits one output sample.
    float acc = 0.0f;
    #pragma unroll 16
    for (int k = 0; k < N; ++k) {
        acc += smem_taps[k] * smem_samples[TILE_HISTORY + tid - k];
    }
    out_c[tid] = acc;

    __syncthreads();
    // Phase 3: tail of new input becomes next call's history (overlap-save).
    for (int j = tid; j < TILE_HISTORY; j += blockDim.x) {
        state_c[j] = smem_samples[win + j];
    }
}

// One-shot, no streaming state. Each (block.x, block.y) pair handles one (sample chunk, channel).
// Boundary is zero (causal FIR — y[0..N-2] are transient).
template <Band B>
__global__ void fir_oneshot_kernel(const float* __restrict__ in,
                                   float*       __restrict__ out,
                                   int n_samp)
{
    constexpr int N = fir_t<B>::N;
    const int ch = blockIdx.y;
    const int t  = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_samp) return;

    const float* in_c = in  + ch * n_samp;
    float        acc  = 0.0f;
    #pragma unroll 16
    for (int k = 0; k < N; ++k) {
        int idx = t - k;
        if (idx >= 0) acc += fir_t<B>::at(k) * in_c[idx];
    }
    out[ch * n_samp + t] = acc;
}

void launch_fir_broadband(const float* in, float* out, float* state,
                          int n_ch, int win, cudaStream_t s)
{
    constexpr int H = NTAPS_BROADBAND - 1;
    constexpr int T = NTAPS_BROADBAND;
    size_t smem_bytes = (H + win + T) * sizeof(float);
    fir_streaming_kernel<BAND_BROADBAND><<<n_ch, win, smem_bytes, s>>>(in, out, state, win);
}

void launch_fir_oneshot(Band band, const float* in, float* out,
                        int n_ch, int n_samp, cudaStream_t s)
{
    constexpr int TPB = 128;
    dim3 grid((n_samp + TPB - 1) / TPB, n_ch);
    dim3 block(TPB);
    switch (band) {
        case BAND_BROADBAND: fir_oneshot_kernel<BAND_BROADBAND><<<grid, block, 0, s>>>(in, out, n_samp); break;
        case BAND_DELTA:     fir_oneshot_kernel<BAND_DELTA>    <<<grid, block, 0, s>>>(in, out, n_samp); break;
        case BAND_THETA:     fir_oneshot_kernel<BAND_THETA>    <<<grid, block, 0, s>>>(in, out, n_samp); break;
        case BAND_ALPHA:     fir_oneshot_kernel<BAND_ALPHA>    <<<grid, block, 0, s>>>(in, out, n_samp); break;
        case BAND_BETA:      fir_oneshot_kernel<BAND_BETA>     <<<grid, block, 0, s>>>(in, out, n_samp); break;
        case BAND_GAMMA:     fir_oneshot_kernel<BAND_GAMMA>    <<<grid, block, 0, s>>>(in, out, n_samp); break;
        case BAND_SSVEP:     fir_oneshot_kernel<BAND_SSVEP>    <<<grid, block, 0, s>>>(in, out, n_samp); break;
        default: break;
    }
}

} // namespace eeg
