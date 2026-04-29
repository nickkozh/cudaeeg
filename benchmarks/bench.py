#!/usr/bin/env python3
"""GPU pipeline vs scipy CPU baselines, swept over 8 / 16 / 32 channels.

Reports per-window latency in ms. Emits a markdown table at the end ready to paste
into the README. Source data: ./data/ if any file is dropped in, otherwise synthetic.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import (butter, sosfilt, sosfilt_zi, sosfiltfilt,
                          iirnotch, lfilter, lfilter_zi, welch, filtfilt)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_loader import load_windows, synth_eeg  # noqa: E402

FS         = 250.0
WIN        = 256
N_WINDOWS  = 240          # ~60 s of data at 250 Hz / 256

_BAND_EDGES = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 100), (5, 20)]


def _band_powers(f: np.ndarray, p: np.ndarray) -> np.ndarray:
    out = np.empty(6, dtype=np.float32)
    for i, (lo, hi) in enumerate(_BAND_EDGES):
        m = (f >= lo) & (f <= hi)
        out[i] = float(np.trapezoid(p[m], f[m])) if m.any() else 0.0
    return out


# ---------------------------------------------------------------------------
# CPU baselines (parameterized by n_ch)
# ---------------------------------------------------------------------------

def _cpu_state(n_ch: int) -> dict:
    sos_bp = butter(4, [0.5, 100.0], btype="bandpass", fs=FS, output="sos")
    b_n, a_n = iirnotch(60.0, 30.0, fs=FS)
    return {
        "sos_bp": sos_bp,
        "zi_bp":  np.repeat(sosfilt_zi(sos_bp)[:, None, :], n_ch, axis=1),
        "b_n":    b_n,
        "a_n":    a_n,
        "zi_n":   np.repeat(lfilter_zi(b_n, a_n)[None, :], n_ch, axis=0),
    }


def cpu_causal(window: np.ndarray, st: dict) -> np.ndarray:
    n_ch = window.shape[0]
    y, st["zi_bp"] = sosfilt(st["sos_bp"], window, axis=1, zi=st["zi_bp"])
    z = np.empty_like(y)
    for c in range(n_ch):
        z[c], st["zi_n"][c] = lfilter(st["b_n"], st["a_n"], y[c], zi=st["zi_n"][c])
    z -= z.mean(axis=0, keepdims=True)
    feats = np.empty((n_ch, 6), dtype=np.float32)
    for c in range(n_ch):
        f, p = welch(z[c], fs=FS, nperseg=128, noverlap=64, window="hann")
        feats[c] = _band_powers(f, p)
    return feats


def cpu_offline(window: np.ndarray) -> np.ndarray:
    n_ch = window.shape[0]
    sos_bp = butter(4, [0.5, 100.0], btype="bandpass", fs=FS, output="sos")
    b_n, a_n = iirnotch(60.0, 30.0, fs=FS)
    y = sosfiltfilt(sos_bp, window, axis=1)
    z = np.array([filtfilt(b_n, a_n, y[c]) for c in range(n_ch)])
    z -= z.mean(axis=0, keepdims=True)
    feats = np.empty((n_ch, 6), dtype=np.float32)
    for c in range(n_ch):
        f, p = welch(z[c], fs=FS, nperseg=128, noverlap=64, window="hann")
        feats[c] = _band_powers(f, p)
    return feats


# ---------------------------------------------------------------------------
# Timed runs
# ---------------------------------------------------------------------------

@dataclass
class Result:
    n_ch: int
    gpu_lat_ms: Optional[np.ndarray]
    cpu_causal_ms: np.ndarray
    cpu_offline_ms: np.ndarray
    ops_per_w_avg: Optional[float]


def _windows(n_ch: int, source_path: str | None) -> np.ndarray:
    """Materialize N_WINDOWS worth of input as a (n_ch, N_WINDOWS*WIN) array."""
    if source_path is not None:
        gen = load_windows(source_path, n_ch=n_ch, win=WIN, fs=FS)
        chunks = [next(gen) for _ in range(N_WINDOWS)]
        return np.concatenate(chunks, axis=1)
    return synth_eeg(n_ch, N_WINDOWS * WIN, FS)


def _time_gpu(n_ch: int, data: np.ndarray) -> tuple[Optional[np.ndarray], Optional[float]]:
    if importlib.util.find_spec("cuda_eeg_prep") is None:
        return None, None
    import cuda_eeg_prep
    from visualizer.power import sample_power_w, estimate_flops_per_window, OpsPerWattTracker

    pipe = cuda_eeg_prep.Pipeline(n_ch, WIN, FS)
    tracker = OpsPerWattTracker(estimate_flops_per_window(n_ch, WIN))
    for _ in range(5):
        pipe.process(data[:, :WIN].astype(np.float32))
    lat = np.empty(N_WINDOWS, dtype=np.float64)
    for w in range(N_WINDOWS):
        chunk = data[:, w * WIN:(w + 1) * WIN].astype(np.float32)
        t0 = time.perf_counter()
        pipe.process(chunk)
        dt = time.perf_counter() - t0
        lat[w] = dt * 1e3
        tracker.update(dt, sample_power_w())
    return lat, tracker.session_avg


def _time_cpu(fn, data: np.ndarray, *args) -> np.ndarray:
    lat = np.empty(N_WINDOWS, dtype=np.float64)
    for w in range(N_WINDOWS):
        chunk = data[:, w * WIN:(w + 1) * WIN]
        t0 = time.perf_counter()
        fn(chunk, *args)
        lat[w] = (time.perf_counter() - t0) * 1e3
    return lat


def run_for(n_ch: int, source_path: str | None) -> Result:
    print(f"\n# n_ch = {n_ch}")
    data = _windows(n_ch, source_path)

    gpu_lat, opw = _time_gpu(n_ch, data)
    if gpu_lat is None:
        print("  [GPU] cuda_eeg_prep not importable — skipping GPU run")
    cpu_c = _time_cpu(cpu_causal, data, _cpu_state(n_ch))
    cpu_o = _time_cpu(cpu_offline, data)
    return Result(n_ch=n_ch, gpu_lat_ms=gpu_lat, cpu_causal_ms=cpu_c,
                  cpu_offline_ms=cpu_o, ops_per_w_avg=opw)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _stat(arr: Optional[np.ndarray], q: float) -> str:
    if arr is None or len(arr) == 0:
        return "—"
    return f"{np.percentile(arr, q):.3f}"


def emit_table(results: list[Result]) -> str:
    lines = []
    lines.append("| n_ch | GPU median | GPU p95 | sosfilt med | sosfiltfilt med | ops/W |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for r in results:
        opw = f"{r.ops_per_w_avg:.2e}" if r.ops_per_w_avg else "—"
        lines.append(
            f"| {r.n_ch} | {_stat(r.gpu_lat_ms, 50)} | {_stat(r.gpu_lat_ms, 95)} | "
            f"{_stat(r.cpu_causal_ms, 50)} | {_stat(r.cpu_offline_ms, 50)} | {opw} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=str, default="8,16,32",
                        help="Comma-separated channel counts to benchmark")
    parser.add_argument("--file", type=str, default=None,
                        help="Optional input file (CSV/EDF/NPY); else synthetic")
    args = parser.parse_args()

    counts = [int(x) for x in args.channels.split(",")]
    results = [run_for(n, args.file) for n in counts]

    print("\n## Benchmark summary (ms per window)")
    print(emit_table(results))


if __name__ == "__main__":
    main()
