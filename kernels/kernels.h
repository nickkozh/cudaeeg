#pragma once
// Internal launcher declarations — not part of the public C API.

#include <cuda_runtime.h>
#include "../include/cuda_eeg_prep.h"

namespace eeg {

// Streaming broadband FIR — one block per channel, WIN threads. State updated in place.
//   in:    [n_ch * win]
//   out:   [n_ch * win]
//   state: [n_ch * (NTAPS_BROADBAND - 1)]
void launch_fir_broadband(const float* in, float* out, float* state,
                          int n_ch, int win, cudaStream_t s);

// One-shot narrowband FIR for the standalone API. Boundary handled as zero history.
void launch_fir_oneshot(Band band, const float* in, float* out,
                        int n_ch, int n_samp, cudaStream_t s);

// IIR notch (60 Hz unless NOTCH_FREQ_HZ overridden). One thread per channel.
//   x:  [n_ch * n_samp] — modified in place
//   z1: [n_ch], z2: [n_ch] — biquad state, persisted across windows
void launch_notch(float* x, float* z1, float* z2, int n_ch, int n_samp, cudaStream_t s);

// CAR — subtract per-sample mean across channels. In place.
void launch_car(float* x, int n_ch, int n_samp, cudaStream_t s);

// Welch PSD: 3 segments of 128 with 50% overlap, Hann window. Writes 6 band powers per channel.
//   in:       [n_ch * win]   (win must be 256)
//   out:      [n_ch * EEG_N_BANDS]
//   fft_buf:  [n_ch * 3 * 128]      — windowed segments, R2C input
//   spec_buf: [n_ch * 3 * 65 * 2]   — cuFFT R2C output (complex)
//   psd_buf:  [n_ch * 65]           — averaged squared magnitude
//   plan:     pre-built cufftHandle (R2C, length 128, batch n_ch * 3)
void launch_psd(const float* in, float* out,
                float* fft_buf, void* spec_buf, float* psd_buf,
                int n_ch, int win, float fs, void* plan, cudaStream_t s);

// Upload Hann window + Welch normalization (1 / (fs * sum(w^2))) to constant memory.
// Called once at pipeline create-time.
void psd_upload_constants(float fs, cudaStream_t s);

} // namespace eeg
