"""Drop-folder data ingestion. Yields (n_ch, win) float32 windows for the pipeline.

Format detection by extension:
    .csv  → np.loadtxt (rows = samples, cols = channels — same as scipy / pandas convention)
    .edf  → pyEDFlib.highlevel.read_edf (header carries fs)
    .npy  → np.load (expected layout: (n_ch, n_samp) channel-major)

If `path` is None, scan ./data/ for the newest file. If no files exist, fall back to
synthetic 1/f noise + 10 Hz alpha + 60 Hz mains.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

DATA_DIR  = Path(__file__).parent / "data"
DEFAULT_FS = 250.0
SUPPORTED  = {".csv", ".edf", ".npy"}


# ---------------------------------------------------------------------------
# Synthetic fallback (also used by bench.py)
# ---------------------------------------------------------------------------

def synth_eeg(n_ch: int, n_samp: int, fs: float = DEFAULT_FS, seed: int = 0xEEEEEE) -> np.ndarray:
    """Pink-ish noise + 10 Hz alpha (chs 0-1) + 60 Hz line noise everywhere."""
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n_samp, d=1.0 / fs)
    spec_shape = np.where(freqs > 0, 1.0 / np.sqrt(freqs + 1e-3), 1.0)
    out = np.empty((n_ch, n_samp), dtype=np.float32)
    t = np.arange(n_samp) / fs
    for c in range(n_ch):
        white = rng.standard_normal(n_samp).astype(np.float32)
        spec = np.fft.rfft(white) * spec_shape
        sig = 5.0 * np.fft.irfft(spec, n=n_samp).astype(np.float32)
        if c < 2:
            sig += 1.5 * np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        sig += 2.0 * np.sin(2 * np.pi * 60.0 * t).astype(np.float32)
        out[c] = sig
    return out


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> tuple[np.ndarray, float]:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)
    # Convention: rows=samples, cols=channels. Transpose to channel-major.
    if arr.shape[1] > arr.shape[0]:
        log.warning("%s: more cols than rows; assuming already channel-major", path.name)
    else:
        arr = arr.T
    return arr.astype(np.float32, copy=False), DEFAULT_FS


def _load_npy(path: Path) -> tuple[np.ndarray, float]:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected 2D array, got shape {arr.shape}")
    if arr.shape[0] > arr.shape[1]:
        # heuristic: more rows than cols → samples × channels, transpose
        arr = arr.T
    return arr.astype(np.float32, copy=False), DEFAULT_FS


def _load_edf(path: Path) -> tuple[np.ndarray, float]:
    try:
        from pyedflib import highlevel
    except ImportError as e:
        raise RuntimeError("EDF support requires pyEDFlib (pip install pyEDFlib)") from e
    sigs, sig_headers, _ = highlevel.read_edf(str(path))
    fs = float(sig_headers[0]["sample_frequency"])
    arr = np.stack([s.astype(np.float32) for s in sigs], axis=0)
    return arr, fs


_LOADERS = {".csv": _load_csv, ".edf": _load_edf, ".npy": _load_npy}


def _newest_file(d: Path) -> Path | None:
    if not d.exists():
        return None
    candidates = [p for p in d.iterdir() if p.suffix.lower() in SUPPORTED and p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _validate(arr: np.ndarray, n_ch: int, fs: float, expected_fs: float, src: str) -> np.ndarray:
    if arr.shape[0] < n_ch:
        raise ValueError(f"{src}: need {n_ch} channels, file has {arr.shape[0]}")
    if abs(fs - expected_fs) > 0.5:
        log.warning("%s: fs=%.1f Hz, pipeline expects %.1f — proceeding anyway", src, fs, expected_fs)
    return arr[:n_ch]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_windows(
    path: str | Path | None = None,
    *,
    n_ch: int,
    win: int = 256,
    fs: float = DEFAULT_FS,
    loop: bool = True,
) -> Iterator[np.ndarray]:
    """Yield successive (n_ch, win) float32 windows.

    If `path` is None and ./data is empty, yields synthetic windows indefinitely.
    `loop=True` wraps around at end-of-file; otherwise stops when the file is exhausted.
    """
    if path is None:
        f = _newest_file(DATA_DIR)
        if f is None:
            log.info("data/ is empty — using synthetic source")
            yield from _synthetic_stream(n_ch, win, fs)
            return
        path = f

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in _LOADERS:
        raise ValueError(f"unsupported extension {suffix}; expected one of {sorted(SUPPORTED)}")

    arr, file_fs = _LOADERS[suffix](path)
    arr = _validate(arr, n_ch, file_fs, fs, path.name)
    log.info("loaded %s: %d ch × %d samp @ %.1f Hz", path.name, arr.shape[0], arr.shape[1], file_fs)

    n_samp = arr.shape[1]
    i = 0
    while True:
        if i + win > n_samp:
            if not loop:
                return
            i = 0
        yield arr[:, i:i + win].copy()
        i += win


def _synthetic_stream(n_ch: int, win: int, fs: float) -> Iterator[np.ndarray]:
    # Pre-generate a long buffer once, then slide a window through it.
    buf = synth_eeg(n_ch, win * 256, fs)
    n_samp = buf.shape[1]
    i = 0
    while True:
        if i + win > n_samp:
            i = 0
        yield buf[:, i:i + win].copy()
        i += win


# ---------------------------------------------------------------------------
# Drop-folder watcher
# ---------------------------------------------------------------------------

class FolderWatcher:
    """1 Hz poll on data/. Sets `pending` when a new (post-creation) file appears.

    Intended use: visualizer/runner consults `pending` between windows; on True it
    swaps to a new generator from `load_windows(...)`. We don't preempt mid-window.
    """

    def __init__(self, d: Path = DATA_DIR, poll_s: float = 1.0):
        self.d = d
        self.poll_s = poll_s
        self._known: set[str] = set()
        self.pending: Path | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if d.exists():
            self._known = {p.name for p in d.iterdir()}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            if not self.d.exists():
                continue
            for p in self.d.iterdir():
                if p.suffix.lower() not in SUPPORTED or p.name in self._known:
                    continue
                self._known.add(p.name)
                self.pending = p
                log.info("new file detected: %s", p.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    gen = load_windows(n_ch=8, win=256)
    for i, w in enumerate(gen):
        print(f"window {i}: {w.shape} {w.dtype}")
        if i >= 4:
            break
