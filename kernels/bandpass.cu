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

// Streaming FIR. smem holds (N-1) history samples followed by `win` new samples;
// each thread emits one output. After compute, the last (N-1) input samples are
// flushed back to global state for the next call (overlap-save).
template <Band B>
__global__ void fir_streaming_kernel(const float* __restrict__ in,
                                     float*       __restrict__ out,
                                     float*       __restrict__ state,
                                     int win)
{
    constexpr int N = fir_t<B>::N;
    constexpr int H = N - 1;
    extern __shared__ float smem[];   // size: H + win

    const int ch  = blockIdx.x;
    const int tid = threadIdx.x;

    const float* in_c    = in    + ch * win;
    float*       out_c   = out   + ch * win;
    float*       state_c = state + ch * H;

    for (int j = tid; j < H + win; j += blockDim.x) {
        smem[j] = (j < H) ? state_c[j] : in_c[j - H];
    }
    __syncthreads();

    float acc = 0.0f;
    #pragma unroll 16
    for (int k = 0; k < N; ++k) {
        acc += fir_t<B>::at(k) * smem[H + tid - k];
    }
    out_c[tid] = acc;

    __syncthreads();
    // Save tail of new input as next call's history.
    for (int j = tid; j < H; j += blockDim.x) {
        state_c[j] = smem[win + j];
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
    size_t smem_bytes = (H + win) * sizeof(float);
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
