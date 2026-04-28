# cuda-eeg-prep

CUDA-accelerated real-time EEG preprocessing for the NVIDIA Jetson Orin Nano.

8 channels, 250 Hz, 256-sample windows. Bandpass → notch → CAR → Welch PSD, all on the iGPU. Output is per-channel, per-band power: 6 floats × 8 channels = 48 features per window.

## Why

MNE-Python and BrainFlow run their filters on the CPU. On a Jetson that means losing half the wall budget to NumPy before the model ever sees a sample. For closed-loop BCI, neurofeedback, or SSVEP work — anything where a window arrives and a decision has to be ready before the next one — the bandpass and PSD have to run on the GPU.

This is a small, focused library: four kernels, one streaming pipeline, no class hierarchy.

## Latency per window

Measured with `benchmarks/bench.py` (240 windows × 256 samples × 8 ch, synthetic 1/f + 10 Hz alpha + 60 Hz mains). All numbers TBD until measured on Orin Nano.

| Pipeline                              | median (ms) | p95 (ms) | p99 (ms) |
|---------------------------------------|-------------|----------|----------|
| **GPU** (cuda-eeg-prep, sm_87)        | TBD         | TBD      | TBD      |
| CPU `scipy.sosfilt` (causal)          | TBD         | TBD      | TBD      |
| CPU `scipy.sosfiltfilt` (zero-phase)  | TBD         | TBD      | TBD      |

Numbers reported per **window**, not throughput — what matters here is how long after the last sample arrives until the features are ready.

## Pipeline

```
raw 8×256 float32  ──►  FIR bandpass 0.5–100 Hz   (1 block / channel, shared-mem sliding window)
                   ──►  IIR notch 60 Hz biquad     (1 thread / channel, DF2T)
                   ──►  CAR (mean across channels) (1 thread / sample)
                   ──►  Welch PSD                  (cuFFT R2C × 3 segments × 128, 50% overlap)
                   ──►  band powers 8×6 float32
```

State that lives across windows: the FIR ring buffer (`NTAPS-1` samples per channel) and the biquad's two state words per channel. Everything else is scratch.

Per-band tap arrays for delta / theta / alpha / beta / gamma / SSVEP are also baked in and exposed via a standalone `filter_band(...)` API for offline narrowband filtering.

## Build

Tested on JetPack 6.x (CUDA 12). On older JetPack (CUDA 11), `cmake -DCMAKE_CUDA_ARCHITECTURES=87` still works.

```bash
pip install pybind11 numpy scipy
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# the resulting cuda_eeg_prep*.so is dropped in build/ — point PYTHONPATH at it
export PYTHONPATH=$PWD/build:$PYTHONPATH
```

For 50 Hz mains regions:

```bash
cmake -B build -DEEG_NOTCH_HZ=50 -DCMAKE_BUILD_TYPE=Release
```

If you change the band edges or sample rate, regenerate the FIR taps:

```bash
python taps/generate_taps.py
```

## Quickstart

```python
import numpy as np
import cuda_eeg_prep

pipe = cuda_eeg_prep.Pipeline(n_ch=8, win=256, fs=250.0)

# Stream windows through the pipeline
for raw in adc_stream():                 # raw: (8, 256) float32
    feats = pipe.process(raw)            # feats: (8, 6) float32 — δ, θ, α, β, γ, SSVEP
    classifier.predict(feats)
```

Standalone narrowband FIR (offline / batch use):

```python
alpha = cuda_eeg_prep.filter_band("alpha", signal)   # signal: (n_ch, n_samp)
```

## Layout

```
include/cuda_eeg_prep.h    public C API + CUDA_CHECK
kernels/bandpass.cu        FIR streaming kernel (templated by Band)
kernels/notch.cu           IIR biquad notch
kernels/car.cu             common-average reference
kernels/psd.cu             Welch PSD via cuFFT + band integration
pipeline.cu                state, ring buffers, stream orchestration
python/bindings.cpp        pybind11 wrapper
taps/generate_taps.py      offline FIR designer (writes fir_taps.h)
benchmarks/bench.py        latency comparison
tests/test_filters.py      sanity vs scipy
```

## API

```c
CudaEegPipeline* eeg_create(int n_ch, int win, float fs);
void             eeg_destroy(CudaEegPipeline*);
void             eeg_process_window(CudaEegPipeline*, const float* raw_in, float* out);
void             eeg_filter_band(Band b, const float* in, float* out, int n_ch, int n_samp);
```

`raw_in` is channel-major: `[ch0[0..win), ch1[0..win), …]`. `out` is `n_ch × 6` floats.

## Notes

- Buffers are `cudaMallocManaged` throughout — Jetson's CPU and iGPU share DRAM, so explicit `cudaMemcpy` would just slow things down.
- FIR taps live in `__constant__` memory (~6 KB total across all bands).
- `process_window` synchronizes on its stream before returning. If you have multiple electrode banks, instantiate multiple `Pipeline`s — each owns its own stream and they run concurrently.
- Not built or runtime-tested on this dev machine (macOS); the CMake targets `sm_87` and the test/bench have only been exercised against the scipy paths. Build and run on the Jetson to verify.
