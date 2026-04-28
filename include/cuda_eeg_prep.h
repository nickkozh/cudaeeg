#pragma once
// cuda-eeg-prep — public C API.
// All buffers passed in/out are float32, managed memory (cudaMallocManaged).
// Caller can pass plain host pointers from numpy too — pipeline copies into managed mem internally.

#include <cstdio>
#include <cstdlib>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BAND_BROADBAND = 0,
    BAND_DELTA     = 1,
    BAND_THETA     = 2,
    BAND_ALPHA     = 3,
    BAND_BETA      = 4,
    BAND_GAMMA     = 5,
    BAND_SSVEP     = 6,
    BAND_COUNT
} Band;

// 6 features per channel: power in delta, theta, alpha, beta, gamma, ssvep.
#define EEG_N_BANDS 6

typedef struct CudaEegPipeline CudaEegPipeline;

CudaEegPipeline* eeg_create(int n_ch, int win, float fs);
void             eeg_destroy(CudaEegPipeline* p);

// raw_in : n_ch * win float32 (channel-major: ch0[0..win), ch1[0..win), ...)
// out    : n_ch * EEG_N_BANDS float32 — band powers per channel.
void eeg_process_window(CudaEegPipeline* p, const float* raw_in, float* out);

// Stateless narrowband FIR. For one-shot or batch use, not streaming —
// caller appends history themselves. n_samp must include any history padding.
void eeg_filter_band(Band band, const float* in, float* out, int n_ch, int n_samp);

#ifdef __cplusplus
}
#endif

// Used at allocation/plan-creation sites and after final stream sync. Not on every kernel launch —
// kernel errors surface at the next sync point regardless.
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                               \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                     #call, __FILE__, __LINE__, cudaGetErrorString(_e));    \
        std::abort();                                                       \
    }                                                                       \
} while (0)
#endif
