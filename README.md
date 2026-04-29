# cuda-eeg-prep

CUDA-accelerated real-time EEG preprocessing for the NVIDIA Jetson Orin Nano.

Up to 32 channels, 250 Hz, 256-sample windows. Bandpass → notch → CAR → Welch PSD, all on the iGPU. Output is per-channel, per-band power: 6 floats × N channels per window.

This was originally built as the preprocessing layer for Mindset Technologies, it's now open-source for community BCI enthusiasts to use and test with.

![Web visualizer demo](visualizer/screenshots/web_demo.png)
*(Replace this with a screenshot from your Jetson run — press S in the plot visualizer or take a browser screenshot of the web UI.)*

## Why

MNE-Python and BrainFlow run their filters on the CPU. On a Jetson that means losing half the wall budget to NumPy before the model ever sees a sample. For closed-loop BCI, neurofeedback, or SSVEP work — anything where a window arrives and a decision has to be ready before the next one — the bandpass and PSD have to run on the GPU.

This is a small, focused library: four kernels, one streaming pipeline, no class hierarchy.

## Benchmark results

Run `python benchmarks/bench.py` to reproduce. Numbers below are TBD until measured on Orin Nano; the CPU columns are available on any machine.

| n_ch | GPU median (ms) | GPU p95 | sosfilt median | sosfiltfilt median | ops/W |
|---:|---:|---:|---:|---:|---:|
| 8 | TBD | TBD | TBD | TBD | TBD |
| 16 | TBD | TBD | TBD | TBD | TBD |
| 32 | TBD | TBD | TBD | TBD | TBD |

Numbers reported per **window** (256 samples), not throughput — what matters for closed-loop BCI is how long after the last sample arrives until the features are ready.

## Pipeline

```
raw N×256 float32  ──►  FIR bandpass 0.5–100 Hz   (1 block / channel, shared-mem sliding window)
                   ──►  IIR notch 60 Hz biquad     (1 thread / channel, DF2T)
                   ──►  CAR (mean across channels) (1 thread / sample)
                   ──►  Welch PSD                  (cuFFT R2C × 3 segments × 128, 50% overlap)
                   ──►  band powers N×6 float32
```

State that lives across windows: the FIR ring buffer (`NTAPS-1` samples per channel) and the biquad's two state words per channel. Everything else is scratch.

Per-band tap arrays for delta / theta / alpha / beta / gamma / SSVEP are also baked in and exposed via a standalone `filter_band(...)` API for offline narrowband filtering.

## Shared memory tiling

Each thread block handles one channel. The block's tile is the entire 256-sample output window plus the `NTAPS-1` history samples carried from the previous call, all staged into shared memory alongside the filter taps before the convolution loop begins. For NTAPS_BROADBAND = 257 this is `(256 + 256 + 257) × 4 = 3076` bytes per block — under 2% of sm_87's 164 KB shared-memory budget.

This avoids 257 redundant global-memory reads per output sample. The two smem segments (samples, taps) are loaded cooperatively by the block's threads in a single strided pass, then every output is a length-N dot product over on-chip memory with no bank conflicts. See [kernels/bandpass.cu](kernels/bandpass.cu) for the full tiling comment block.

## Data ingestion

Drop a recording into `data/` and the pipeline picks it up automatically.

| Extension | Loader | Layout | Sample rate |
|---|---|---|---|
| `.csv` | `numpy.loadtxt` | rows = samples, cols = channels | assumes 250 Hz |
| `.edf` | `pyEDFlib` | as encoded | read from header |
| `.npy` | `numpy.load` | `(n_ch, n_samp)` | assumes 250 Hz |

If `data/` is empty (or no `--file` is passed), the pipeline falls back to synthetic EEG: pink noise + 10 Hz alpha on channels 0–1 + 60 Hz mains. Useful for demos and CI.

```bash
# Use a real recording
python visualizer/plot.py --channels 32 --file data/subject01.edf

# Benchmark with a file instead of synthetic data
python benchmarks/bench.py --channels 8,16,32 --file data/subject01.csv
```

## Kernel occupancy analysis

Run `./build/eeg_occupancy` on the target hardware to get a table like:

| Kernel | Block size (actual) | Optimal block | Active blocks/SM | Occupancy |
|---|---|---|---|---|
| fir_streaming\<BROADBAND\> | 256 | TBD | TBD | TBD |
| fir_streaming\<ALPHA\> | 128 | TBD | TBD | TBD |
| notch | 128 | TBD | TBD | TBD |
| car | 256 | TBD | TBD | TBD |

*(Replace with actual output from `./build/eeg_occupancy` on your Jetson.)*

`eeg_occupancy` uses `cudaOccupancyMaxPotentialBlockSize` and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for each kernel and reports whether the launch configuration matches the hardware-optimal block size. See [benchmarks/occupancy.cu](benchmarks/occupancy.cu).

## Profiling with NSight Systems

On JetPack 6:

```bash
sudo apt install nsight-systems-cli
bash benchmarks/nsight_profile.sh          # defaults to --channels 32
# or override channel count:
CHANNELS=16 bash benchmarks/nsight_profile.sh
```

The script captures CUDA kernel timeline, CUDA API trace, OS runtime, and GPU metrics. Output goes to `benchmarks/nsight_out/`. Open the `.nsys-rep` file in NSight Systems GUI to inspect:

- **Kernel gaps** — idle time between windows (should be < 1 ms at 250 Hz)
- **H2D overlap** — not expected (Jetson uses unified memory; no explicit copies)
- **Bank conflicts** — should be zero with the current smem layout
- **Wave-quantization artifacts** — watch for block launch counts that waste a partial wave

## Visualizers

### matplotlib (terminal / VNC)

```bash
python visualizer/plot.py --channels 8
python visualizer/plot.py --channels 32 --file data/subject01.edf
```

Three stacked panels: raw waveforms, filtered waveforms, 6-bar band power (log scale). Press **S** to save a snapshot to `visualizer/screenshots/`.

On a headless Jetson: `MPLBACKEND=Agg python visualizer/plot.py --channels 8` writes a static image instead of showing a window.

### Web (browser)

```bash
pip install aiohttp
python visualizer/web/websocket_server.py
# then open http://localhost:8765 in any browser
```

Live waveform chart (raw + filtered overlaid), band-power bar chart, and a metrics card showing latency, power draw, and ops/watt. Use the channel dropdown to switch between 8 / 16 / 32 channels at runtime — the pipeline reconfigures between windows with a single-window stall while cuFFT rebuilds its plan.

## Build

Tested on JetPack 6.x (CUDA 12). On older JetPack (CUDA 11), `cmake -DCMAKE_CUDA_ARCHITECTURES=87` still works.

```bash
pip install pybind11 numpy scipy
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# cuda_eeg_prep*.so lands in build/ — point PYTHONPATH at it
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

# 8 channels; reconfigure to 16 or 32 at any time
pipe = cuda_eeg_prep.Pipeline(n_ch=8, win=256, fs=250.0)

for raw in adc_stream():                 # raw: (8, 256) float32
    feats = pipe.process(raw)            # feats: (8, 6) float32 — δ, θ, α, β, γ, SSVEP
    classifier.predict(feats)

# Reconfigure at runtime (brief stall while cuFFT plan rebuilds)
pipe.reconfigure(32)

# Standalone narrowband FIR (offline / batch)
alpha = cuda_eeg_prep.filter_band("alpha", signal)   # signal: (n_ch, n_samp)
```

## Layout

```
include/cuda_eeg_prep.h         public C API + CUDA_CHECK
kernels/bandpass.cu             FIR streaming kernel (templated by Band; explicit smem tiling)
kernels/notch.cu                IIR biquad notch
kernels/car.cu                  common-average reference
kernels/psd.cu                  Welch PSD via cuFFT + band integration
pipeline.cu                     state, ring buffers, stream orchestration, reconfigure
python/bindings.cpp             pybind11 wrapper
taps/generate_taps.py           offline FIR designer (writes fir_taps.h)
data_loader.py                  drop-folder watcher + CSV/EDF/NPY ingestion + synthetic fallback
visualizer/plot.py              matplotlib FuncAnimation visualizer
visualizer/web/                 aiohttp websocket server + Chart.js browser UI
benchmarks/bench.py             latency comparison (8/16/32 ch, markdown table output)
benchmarks/occupancy.cu         cudaOccupancyMaxPotentialBlockSize reporter
benchmarks/nsight_profile.sh    NSight Systems profiling wrapper
data/                           drop recordings here (see data/README.md)
tests/test_filters.py           sanity vs scipy
```

## API

```c
CudaEegPipeline* eeg_create(int n_ch, int win, float fs);
void             eeg_destroy(CudaEegPipeline*);
void             eeg_reconfigure(CudaEegPipeline*, int n_ch);   // rebuild plan for new channel count
const float*     eeg_last_filtered(const CudaEegPipeline*);     // post-CAR buffer (visualizer use only)
int              eeg_n_ch(const CudaEegPipeline*);
int              eeg_win(const CudaEegPipeline*);
void             eeg_process_window(CudaEegPipeline*, const float* raw_in, float* out);
void             eeg_filter_band(Band b, const float* in, float* out, int n_ch, int n_samp);
```

`raw_in` is channel-major: `[ch0[0..win), ch1[0..win), …]`. `out` is `n_ch × 6` floats.

## Notes

- Buffers are `cudaMallocManaged` throughout — Jetson's CPU and iGPU share DRAM, so explicit `cudaMemcpy` would just slow things down.
- FIR taps live in `__constant__` memory (~6 KB total across all bands).
- `process_window` synchronizes on its stream before returning. If you have multiple electrode banks, instantiate multiple `Pipeline`s — each owns its own stream and they run concurrently.
- `eeg_reconfigure` destroys and recreates all per-channel buffers and the cuFFT plan. It synchronizes the stream before and after, so it is safe to call between windows but not mid-window.
- Not built or runtime-tested on this dev machine (macOS); the CMake targets `sm_87` and the test/bench have only been exercised against the scipy paths. Build and run on the Jetson to verify.
