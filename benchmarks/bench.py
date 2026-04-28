#!/usr/bin/env python3
# GPU pipeline vs scipy.sosfilt (causal, real-time-fair) and sosfiltfilt (zero-phase reference).
# Reports per-window latency in ms — not throughput.

from __future__ import annotations
import time
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, iirnotch, lfilter, lfilter_zi, welch

FS         = 250.0
N_CH       = 8
WIN        = 256
N_WINDOWS  = 240          # ~60 s of data at 250 Hz / 256
RNG_SEED   = 0xEEEEEE


def synth_eeg(n_ch: int, n_samp: int, fs: float, seed: int) -> np.ndarray:
    """1/f pink-ish noise + 10 Hz alpha on chs 0–1 + 60 Hz line noise everywhere."""
    rng = np.random.default_rng(seed)
    # Pink noise via spectrum shaping
    n_fft = n_samp
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    spec_shape = np.where(freqs > 0, 1.0 / np.sqrt(freqs + 1e-3), 1.0)
    out = np.empty((n_ch, n_samp), dtype=np.float32)
    t = np.arange(n_samp) / fs
    for c in range(n_ch):
        white = rng.standard_normal(n_fft).astype(np.float32)
        spec = np.fft.rfft(white) * spec_shape
        pink = np.fft.irfft(spec, n=n_fft).astype(np.float32)
        sig = 5.0 * pink
        if c < 2:
            sig += 1.5 * np.sin(2 * np.pi * 10.0 * t).astype(np.float32)   # alpha
        sig += 2.0 * np.sin(2 * np.pi * 60.0 * t).astype(np.float32)        # mains
        out[c] = sig
    return out


# ---------- CPU baselines ----------

def cpu_state_factory():
    sos_bp = butter(4, [0.5, 100.0], btype="bandpass", fs=FS, output="sos")
    b_n, a_n = iirnotch(60.0, 30.0, fs=FS)
    return {
        "sos_bp": sos_bp,
        "zi_bp":  np.repeat(sosfilt_zi(sos_bp)[:, None, :], N_CH, axis=1),  # per-channel state
        "b_n":    b_n,
        "a_n":    a_n,
        "zi_n":   np.repeat(lfilter_zi(b_n, a_n)[None, :], N_CH, axis=0),
    }


def cpu_causal_window(win: np.ndarray, st: dict) -> np.ndarray:
    y, st["zi_bp"] = sosfilt(st["sos_bp"], win, axis=1, zi=st["zi_bp"])
    z = np.empty_like(y)
    for c in range(N_CH):
        z[c], st["zi_n"][c] = lfilter(st["b_n"], st["a_n"], y[c], zi=st["zi_n"][c])
    z -= z.mean(axis=0, keepdims=True)
    feats = np.empty((N_CH, 6), dtype=np.float32)
    for c in range(N_CH):
        f, p = welch(z[c], fs=FS, nperseg=128, noverlap=64, window="hann")
        feats[c] = _band_powers(f, p)
    return feats


def cpu_offline_window(win: np.ndarray) -> np.ndarray:
    from scipy.signal import filtfilt
    sos_bp = butter(4, [0.5, 100.0], btype="bandpass", fs=FS, output="sos")
    b_n, a_n = iirnotch(60.0, 30.0, fs=FS)
    # filtfilt requires more samples than filter order — for 256 samples this works.
    y = sosfiltfilt(sos_bp, win, axis=1)
    z = np.array([filtfilt(b_n, a_n, y[c]) for c in range(N_CH)])
    z -= z.mean(axis=0, keepdims=True)
    feats = np.empty((N_CH, 6), dtype=np.float32)
    for c in range(N_CH):
        f, p = welch(z[c], fs=FS, nperseg=128, noverlap=64, window="hann")
        feats[c] = _band_powers(f, p)
    return feats


_BAND_EDGES = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 100), (5, 20)]


def _band_powers(f: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.empty(6, dtype=np.float32)
    for i, (lo, hi) in enumerate(_BAND_EDGES):
        m = (f >= lo) & (f <= hi)
        out[i] = float(np.trapezoid(p[m], f[m])) if m.any() else 0.0
    return out


# ---------- GPU pipeline ----------

def time_gpu(data: np.ndarray) -> np.ndarray:
    try:
        import cuda_eeg_prep
    except ImportError:
        print("[GPU] cuda_eeg_prep not built — skipping. Build with cmake first.")
        return np.array([])
    pipe = cuda_eeg_prep.Pipeline(N_CH, WIN, FS)
    # Warmup
    for _ in range(5):
        pipe.process(data[:, :WIN].astype(np.float32))
    lat = []
    for w in range(N_WINDOWS):
        chunk = data[:, w * WIN:(w + 1) * WIN].astype(np.float32)
        t0 = time.perf_counter()
        _ = pipe.process(chunk)
        lat.append((time.perf_counter() - t0) * 1e3)
    return np.array(lat)


def time_cpu_causal(data: np.ndarray) -> np.ndarray:
    st = cpu_state_factory()
    lat = []
    for w in range(N_WINDOWS):
        chunk = data[:, w * WIN:(w + 1) * WIN]
        t0 = time.perf_counter()
        _ = cpu_causal_window(chunk, st)
        lat.append((time.perf_counter() - t0) * 1e3)
    return np.array(lat)


def time_cpu_offline(data: np.ndarray) -> np.ndarray:
    lat = []
    for w in range(N_WINDOWS):
        chunk = data[:, w * WIN:(w + 1) * WIN]
        t0 = time.perf_counter()
        _ = cpu_offline_window(chunk)
        lat.append((time.perf_counter() - t0) * 1e3)
    return np.array(lat)


def fmt(name: str, lat: np.ndarray) -> str:
    if len(lat) == 0:
        return f"{name:<24s}   not run"
    return (f"{name:<24s}   median {np.median(lat):6.3f}   "
            f"p95 {np.percentile(lat, 95):6.3f}   "
            f"p99 {np.percentile(lat, 99):6.3f}   ms / window")


def main() -> None:
    print(f"# Synthesizing {N_WINDOWS} windows × {WIN} samples × {N_CH} ch @ {FS} Hz")
    data = synth_eeg(N_CH, N_WINDOWS * WIN, FS, RNG_SEED)

    print("# Running benchmarks…\n")
    lat_gpu     = time_gpu(data)
    lat_causal  = time_cpu_causal(data)
    lat_offline = time_cpu_offline(data)

    print("\n## Latency per window (ms)")
    print(fmt("GPU pipeline (Jetson)", lat_gpu))
    print(fmt("CPU sosfilt (causal)",  lat_causal))
    print(fmt("CPU sosfiltfilt (offl)", lat_offline))


if __name__ == "__main__":
    main()
