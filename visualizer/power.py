"""Jetson power telemetry. Returns total board power in watts, or None elsewhere.

Probes (in order):
    1. /sys/bus/i2c/drivers/ina3221*/  — older Jetson firmware (Xavier, Nano TX2)
    2. /sys/class/hwmon/hwmon*/        — Orin family
    3. fall back to None and log a single warning

Sample rate is whatever the caller wants — sysfs reads are a few microseconds.
"""

from __future__ import annotations

import glob
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_path: Optional[Path] = None
_kind: str = ""           # "power_uw" | "v_mv_x_i_ma" | "unavailable"
_v_path: Optional[Path] = None
_i_path: Optional[Path] = None
_warned = False


def _probe() -> None:
    global _path, _kind, _v_path, _i_path, _warned

    # INA3221 — power directly in microwatts.
    for cand in glob.glob("/sys/bus/i2c/drivers/ina3221*/[0-9]*/iio:device*/in_power0_input"):
        _path, _kind = Path(cand), "power_uw"
        log.info("power: ina3221 → %s", cand)
        return
    for cand in glob.glob("/sys/bus/i2c/drivers/ina3221*/[0-9]*/hwmon/hwmon*/power1_input"):
        _path, _kind = Path(cand), "power_uw"
        log.info("power: ina3221 hwmon → %s", cand)
        return

    # Generic hwmon — V × I.
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        v = Path(hw) / "in1_input"        # millivolts
        i = Path(hw) / "curr1_input"      # milliamps
        if v.exists() and i.exists():
            _v_path, _i_path = v, i
            _kind = "v_mv_x_i_ma"
            log.info("power: hwmon V×I → %s", hw)
            return

    _kind = "unavailable"
    if not _warned:
        log.warning("power: no INA3221 or hwmon source — telemetry will be None")
        _warned = True


def sample_power_w() -> Optional[float]:
    """Return total board power in watts, or None if not on Jetson."""
    if _kind == "":
        _probe()
    try:
        if _kind == "power_uw":
            return int(_path.read_text().strip()) / 1_000_000.0
        if _kind == "v_mv_x_i_ma":
            mv = int(_v_path.read_text().strip())
            ma = int(_i_path.read_text().strip())
            return (mv * ma) / 1_000_000.0
    except (OSError, ValueError):
        return None
    return None


# ---------------------------------------------------------------------------
# FLOPs estimator — used to compute ops/watt
# ---------------------------------------------------------------------------

def estimate_flops_per_window(n_ch: int, win: int = 256, ntaps: int = 257,
                              n_seg: int = 3, seg: int = 128, n_bins: int = 65) -> float:
    """Rough op count per process_window. Multiplications + additions = 2 each in MAC."""
    bandpass = n_ch * win * (2 * ntaps)              # FIR MACs
    notch    = n_ch * win * 5                         # biquad: 5 mul, 4 add ≈ 9; round
    car      = n_ch * win * 2                         # sum + sub
    # PSD: FFT ~ 5 * N * log2(N) per transform (radix-2 estimate); then magsq + reduction
    import math
    fft = n_ch * n_seg * (5 * seg * math.log2(seg))
    psd = n_ch * n_bins * 4                           # |X|² + scale + accumulate
    return float(bandpass + notch + car + fft + psd)


class OpsPerWattTracker:
    """Rolling and session ops-per-watt computed from per-window FLOP count + power sample."""

    def __init__(self, flops_per_window: float):
        self.flops_per_window = flops_per_window
        self._ema_alpha = 0.2
        self._ema = 0.0
        self._sum = 0.0
        self._n = 0

    def update(self, latency_s: float, power_w: float | None) -> tuple[float | None, float | None]:
        if power_w is None or power_w <= 0 or latency_s <= 0:
            return None, None
        flops_per_s = self.flops_per_window / latency_s
        ops_per_w = flops_per_s / power_w
        self._sum += ops_per_w
        self._n += 1
        self._ema = ops_per_w if self._n == 1 else (self._ema_alpha * ops_per_w + (1 - self._ema_alpha) * self._ema)
        avg = self._sum / self._n
        return ops_per_w, avg

    @property
    def session_avg(self) -> float | None:
        return self._sum / self._n if self._n else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t0 = time.perf_counter()
    for _ in range(3):
        print("power_w =", sample_power_w())
        time.sleep(0.2)
    print("elapsed:", time.perf_counter() - t0)
