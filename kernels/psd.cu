#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include "kernels.h"

namespace eeg {

constexpr int SEG = 128;          // FFT length per Welch segment
constexpr int NSEG = 3;           // 3 segments × 128 with 50% overlap covers a 256-sample window
constexpr int NBINS = SEG / 2 + 1;

// Standard EEG band edges (Hz). Order matches index 0..5 returned by the pipeline.
__constant__ float BAND_LO[EEG_N_BANDS] = { 0.5f,  4.0f,  8.0f, 13.0f, 30.0f,  5.0f };
__constant__ float BAND_HI[EEG_N_BANDS] = { 4.0f,  8.0f, 13.0f, 30.0f,100.0f, 20.0f };
// (delta, theta, alpha, beta, gamma, ssvep)

__constant__ float HANN[SEG];
__constant__ float HANN_NORM;     // 1 / (fs * sum(hann^2)); set at pipeline create-time

__global__ void segment_window_kernel(const float* in, float* fft_buf, int win)
{
    int s   = blockIdx.x;
    int ch  = blockIdx.y;
    int t   = threadIdx.x;
    int off = s * (SEG / 2);                         // 50% overlap
    fft_buf[(ch * NSEG + s) * SEG + t] = in[ch * win + off + t] * HANN[t];
}

__global__ void magsq_avg_kernel(const float2* spec, float* psd)
{
    int b  = blockIdx.x * blockDim.x + threadIdx.x;
    int ch = blockIdx.y;
    if (b >= NBINS) return;

    float acc = 0.0f;
    #pragma unroll
    for (int s = 0; s < NSEG; ++s) {
        float2 v = spec[(ch * NSEG + s) * NBINS + b];
        acc += v.x * v.x + v.y * v.y;
    }
    psd[ch * NBINS + b] = (acc / (float)NSEG) * HANN_NORM;
}

// One block per channel, EEG_N_BANDS threads (6). Each thread integrates one band.
__global__ void band_power_kernel(const float* psd, float* out, float df)
{
    int ch = blockIdx.x;
    int b  = threadIdx.x;
    if (b >= EEG_N_BANDS) return;

    int lo = (int)ceilf(BAND_LO[b] / df);
    int hi = (int)floorf(BAND_HI[b] / df);
    if (lo < 1)       lo = 1;
    if (hi >= NBINS)  hi = NBINS - 1;

    float pwr = 0.0f;
    for (int k = lo; k <= hi; ++k) {
        // One-sided spectrum: double interior bins, leave DC and Nyquist alone.
        float c = (k == 0 || k == NBINS - 1) ? 1.0f : 2.0f;
        pwr += c * psd[ch * NBINS + k];
    }
    out[ch * EEG_N_BANDS + b] = pwr * df;
}

// Host helper: upload Hann + its normalization factor. Called once from pipeline ctor.
void psd_upload_constants(float fs, cudaStream_t s)
{
    float hann[SEG];
    float sum_w2 = 0.0f;
    for (int i = 0; i < SEG; ++i) {
        hann[i]  = 0.5f * (1.0f - cosf(2.0f * 3.14159265358979323846f * i / (SEG - 1)));
        sum_w2  += hann[i] * hann[i];
    }
    float norm = 1.0f / (fs * sum_w2);
    cudaMemcpyToSymbolAsync(HANN, hann, sizeof(hann), 0, cudaMemcpyHostToDevice, s);
    cudaMemcpyToSymbolAsync(HANN_NORM, &norm, sizeof(norm), 0, cudaMemcpyHostToDevice, s);
}

void launch_psd(const float* in, float* out,
                float* fft_buf, void* spec_buf, float* psd_buf,
                int n_ch, int win, float fs, void* plan, cudaStream_t s)
{
    cufftHandle p = (cufftHandle)(uintptr_t)plan;
    cufftSetStream(p, s);

    // 1) segment + Hann window
    dim3 g1(NSEG, n_ch);
    segment_window_kernel<<<g1, SEG, 0, s>>>(in, fft_buf, win);

    // 2) batched R2C FFT (length 128, batch n_ch * NSEG)
    cufftExecR2C(p, (cufftReal*)fft_buf, (cufftComplex*)spec_buf);

    // 3) |X|^2, average across segments, scale
    dim3 g2((NBINS + 31) / 32, n_ch);
    magsq_avg_kernel<<<g2, 32, 0, s>>>((const float2*)spec_buf, psd_buf);

    // 4) integrate PSD over each band
    float df = fs / (float)SEG;
    band_power_kernel<<<n_ch, EEG_N_BANDS, 0, s>>>(psd_buf, out, df);
}

} // namespace eeg
