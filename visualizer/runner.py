"""Drives the CUDA pipeline window-by-window. Yields visualization-ready frames.

Used by both the matplotlib and websocket visualizers so they share metric logic.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from data_loader import load_windows
from visualizer.power import OpsPerWattTracker, estimate_flops_per_window, sample_power_w

log = logging.getLogger(__name__)

_HAS_GPU = importlib.util.find_spec("cuda_eeg_prep") is not None
if _HAS_GPU:
    import cuda_eeg_prep  # type: ignore


@dataclass
class Frame:
    raw:          np.ndarray              # (n_ch, win)
    filtered:     np.ndarray              # (n_ch, win)
    band_powers:  np.ndarray              # (n_ch, 6)
    n_ch:         int
    latency_ms:   float
    power_w:      Optional[float]
    ops_per_w:    Optional[float]
    avg_ops_per_w: Optional[float]
    baseline_8ch_throughput: Optional[float] = None
    relative_throughput:     Optional[float] = None


class Runner:
    """One Pipeline + one data source. Reconfigurable channel count.

    If cuda_eeg_prep isn't built, runs in passthrough mode: returns the raw window
    as 'filtered' and zeros for band_powers, so visualizers can still smoke-test on
    macOS without the .so.
    """

    def __init__(self, n_ch: int, fs: float = 250.0, win: int = 256,
                 source_path: str | None = None):
        self.fs = fs
        self.win = win
        self.source_path = source_path
        self._n_ch = n_ch
        self._make_pipeline_and_source(n_ch)
        self._tracker = OpsPerWattTracker(estimate_flops_per_window(n_ch, win))
        self._baseline_8ch_throughput: Optional[float] = None

    def _make_pipeline_and_source(self, n_ch: int):
        if _HAS_GPU:
            self._pipe = cuda_eeg_prep.Pipeline(n_ch, self.win, self.fs)
        else:
            self._pipe = None
        self._source = load_windows(self.source_path, n_ch=n_ch, win=self.win, fs=self.fs)

    def reconfigure(self, n_ch: int) -> None:
        if n_ch == self._n_ch:
            return
        log.info("reconfiguring: %d → %d channels", self._n_ch, n_ch)
        if _HAS_GPU and self._pipe is not None:
            self._pipe.reconfigure(n_ch)
        self._n_ch = n_ch
        # New data generator at the new channel count.
        self._source = load_windows(self.source_path, n_ch=n_ch, win=self.win, fs=self.fs)
        self._tracker = OpsPerWattTracker(estimate_flops_per_window(n_ch, self.win))

    def set_baseline_throughput(self, windows_per_s: float) -> None:
        """Called by bench/runner once the 8ch baseline has been measured this session."""
        self._baseline_8ch_throughput = windows_per_s

    @property
    def n_ch(self) -> int:
        return self._n_ch

    def __iter__(self) -> Iterator[Frame]:
        for raw in self._source:
            t0 = time.perf_counter()
            if self._pipe is not None:
                bands = self._pipe.process(raw)
                filtered = self._pipe.last_filtered()
            else:
                bands = np.zeros((self._n_ch, 6), dtype=np.float32)
                filtered = raw.copy()
            latency_s = time.perf_counter() - t0

            pw = sample_power_w()
            opw, avg_opw = self._tracker.update(latency_s, pw)

            tput = 1.0 / latency_s if latency_s > 0 else None
            rel = (tput / self._baseline_8ch_throughput) if (tput and self._baseline_8ch_throughput) else None

            yield Frame(
                raw=raw,
                filtered=filtered,
                band_powers=bands,
                n_ch=self._n_ch,
                latency_ms=latency_s * 1000.0,
                power_w=pw,
                ops_per_w=opw,
                avg_ops_per_w=avg_opw,
                baseline_8ch_throughput=self._baseline_8ch_throughput,
                relative_throughput=rel,
            )


def metrics_overlay_text(f: Frame) -> str:
    """Compact human-readable overlay used by both visualizers."""
    lines = [
        f"channels: {f.n_ch}",
        f"latency:  {f.latency_ms:6.2f} ms",
    ]
    if f.ops_per_w is not None:
        lines.append(f"ops/W:    {f.ops_per_w:8.2e}")
        lines.append(f"avg/W:    {f.avg_ops_per_w:8.2e}")
    else:
        lines.append("power:    unavailable")
    if f.relative_throughput is not None:
        lines.append(f"vs 8ch:   {f.relative_throughput:5.2f}x")
    return "\n".join(lines)
