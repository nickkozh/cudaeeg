#include <cuda_runtime.h>
#include <cufft.h>
#include <cstring>

#include "include/cuda_eeg_prep.h"
#include "kernels/kernels.h"
#include "taps/fir_taps.h"

// Streaming pipeline state. Unified-memory throughout — Jetson Orin shares CPU/GPU DRAM,
// so cudaMallocManaged + a single stream avoid the host↔device copy cost entirely.
struct CudaEegPipeline {
    int  n_ch;
    int  win;
    float fs;

    cudaStream_t stream;
    cufftHandle  plan;

    // Per-window buffers
    float* raw;        // [n_ch * win] — input scratch (host copies into here)
    float* filtered;   // [n_ch * win] — FIR out, then notch & CAR in place
    float* features;   // [n_ch * EEG_N_BANDS] — output staging

    // Streaming state (lives across calls)
    float* state_fir;  // [n_ch * (NTAPS_BROADBAND - 1)]
    float* z1;         // [n_ch] biquad state
    float* z2;         // [n_ch]

    // PSD scratch
    float* fft_buf;    // [n_ch * 3 * 128]
    float* spec_buf;   // [n_ch * 3 * 65 * 2]  (cufftComplex)
    float* psd_buf;    // [n_ch * 65]
};

static float* alloc_managed(size_t n) {
    float* p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, n * sizeof(float)));
    std::memset(p, 0, n * sizeof(float));
    return p;
}

extern "C" CudaEegPipeline* eeg_create(int n_ch, int win, float fs)
{
    auto* p = new CudaEegPipeline{};
    p->n_ch = n_ch;
    p->win  = win;
    p->fs   = fs;

    CUDA_CHECK(cudaStreamCreate(&p->stream));

    p->raw       = alloc_managed((size_t)n_ch * win);
    p->filtered  = alloc_managed((size_t)n_ch * win);
    p->features  = alloc_managed((size_t)n_ch * EEG_N_BANDS);

    p->state_fir = alloc_managed((size_t)n_ch * (eeg::NTAPS_BROADBAND - 1));
    p->z1        = alloc_managed(n_ch);
    p->z2        = alloc_managed(n_ch);

    p->fft_buf   = alloc_managed((size_t)n_ch * 3 * 128);
    p->spec_buf  = alloc_managed((size_t)n_ch * 3 * (128 / 2 + 1) * 2);
    p->psd_buf   = alloc_managed((size_t)n_ch * (128 / 2 + 1));

    // cuFFT plan: 128-pt real-to-complex, batch = n_ch * NSEG
    cufftResult cr = cufftPlan1d(&p->plan, 128, CUFFT_R2C, n_ch * 3);
    if (cr != CUFFT_SUCCESS) { std::fprintf(stderr, "cufftPlan1d failed: %d\n", cr); std::abort(); }
    cufftSetStream(p->plan, p->stream);

    eeg::psd_upload_constants(fs, p->stream);
    CUDA_CHECK(cudaStreamSynchronize(p->stream));
    return p;
}

extern "C" void eeg_destroy(CudaEegPipeline* p)
{
    if (!p) return;
    cufftDestroy(p->plan);
    cudaFree(p->raw);
    cudaFree(p->filtered);
    cudaFree(p->features);
    cudaFree(p->state_fir);
    cudaFree(p->z1);
    cudaFree(p->z2);
    cudaFree(p->fft_buf);
    cudaFree(p->spec_buf);
    cudaFree(p->psd_buf);
    cudaStreamDestroy(p->stream);
    delete p;
}

extern "C" void eeg_process_window(CudaEegPipeline* p, const float* raw_in, float* out)
{
    const size_t in_bytes  = (size_t)p->n_ch * p->win * sizeof(float);
    const size_t out_bytes = (size_t)p->n_ch * EEG_N_BANDS * sizeof(float);

    // Host pointer in → managed buffer. On Jetson cudaMemcpyAsync over unified mem is effectively a memcpy
    // but we keep it on the stream so subsequent kernels see the data.
    CUDA_CHECK(cudaMemcpyAsync(p->raw, raw_in, in_bytes, cudaMemcpyDefault, p->stream));

    eeg::launch_fir_broadband(p->raw, p->filtered, p->state_fir, p->n_ch, p->win, p->stream);
    eeg::launch_notch(p->filtered, p->z1, p->z2, p->n_ch, p->win, p->stream);
    eeg::launch_car(p->filtered, p->n_ch, p->win, p->stream);
    eeg::launch_psd(p->filtered, p->features,
                    p->fft_buf, p->spec_buf, p->psd_buf,
                    p->n_ch, p->win, p->fs, (void*)(uintptr_t)p->plan, p->stream);

    CUDA_CHECK(cudaMemcpyAsync(out, p->features, out_bytes, cudaMemcpyDefault, p->stream));
    CUDA_CHECK(cudaStreamSynchronize(p->stream));
}

extern "C" void eeg_filter_band(Band band, const float* in, float* out, int n_ch, int n_samp)
{
    size_t bytes = (size_t)n_ch * n_samp * sizeof(float);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMallocManaged(&d_in,  bytes));
    CUDA_CHECK(cudaMallocManaged(&d_out, bytes));
    std::memcpy(d_in, in, bytes);

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    eeg::launch_fir_oneshot(band, d_in, d_out, n_ch, n_samp, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    cudaStreamDestroy(s);

    std::memcpy(out, d_out, bytes);
    cudaFree(d_in);
    cudaFree(d_out);
}
