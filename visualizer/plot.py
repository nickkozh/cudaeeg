"""Matplotlib live visualizer. Three stacked panels + corner metrics overlay.

Press 'S' to save a PNG snapshot to visualizer/screenshots/plot_snapshot.png.

Usage:
    python visualizer/plot.py [--channels 8|16|32] [--file path/to/data.csv]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from visualizer.runner import Runner, metrics_overlay_text

BAND_LABELS = ["δ", "θ", "α", "β", "γ", "SSVEP"]
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=8, choices=[8, 16, 32])
    parser.add_argument("--file", type=str, default=None, help="Optional input file path")
    parser.add_argument("--fs", type=float, default=250.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    runner = Runner(n_ch=args.channels, fs=args.fs, source_path=args.file)
    frames = iter(runner)

    fig, (ax_raw, ax_filt, ax_psd) = plt.subplots(3, 1, figsize=(11, 8))
    fig.suptitle(f"cuda-eeg-prep — {args.channels} channels @ {args.fs:.0f} Hz")

    t = np.arange(runner.win) / runner.fs
    raw_lines  = [ax_raw.plot(t, np.zeros(runner.win), lw=0.7)[0]  for _ in range(runner.n_ch)]
    filt_lines = [ax_filt.plot(t, np.zeros(runner.win), lw=0.7)[0] for _ in range(runner.n_ch)]
    bars = ax_psd.bar(BAND_LABELS, np.zeros(6))

    ax_raw.set_title("Raw input")
    ax_raw.set_xlabel("s")
    ax_filt.set_title("After bandpass + notch + CAR")
    ax_filt.set_xlabel("s")
    ax_psd.set_title("Band power (mean across channels)")
    ax_psd.set_yscale("log")

    overlay = ax_raw.text(
        0.99, 0.97, "", transform=ax_raw.transAxes,
        ha="right", va="top", family="monospace", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.85),
    )

    def on_key(event):
        if event.key in ("s", "S"):
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            out = SCREENSHOT_DIR / "plot_snapshot.png"
            fig.savefig(out, dpi=150)
            print(f"saved {out}")

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        try:
            f = next(frames)
        except StopIteration:
            return [*raw_lines, *filt_lines, *bars, overlay]

        # If channel count changed (reconfigure), rebuild line objects.
        if len(raw_lines) != f.n_ch:
            for ln in raw_lines + filt_lines:
                ln.remove()
            raw_lines.clear(); filt_lines.clear()
            raw_lines.extend(ax_raw.plot(t, np.zeros(runner.win), lw=0.7)[0]  for _ in range(f.n_ch))
            filt_lines.extend(ax_filt.plot(t, np.zeros(runner.win), lw=0.7)[0] for _ in range(f.n_ch))

        for i, ln in enumerate(raw_lines):  ln.set_ydata(f.raw[i])
        for i, ln in enumerate(filt_lines): ln.set_ydata(f.filtered[i])
        for i, b in enumerate(bars):
            b.set_height(max(float(f.band_powers[:, i].mean()), 1e-12))

        ax_raw.relim();  ax_raw.autoscale_view(scaley=True, scalex=False)
        ax_filt.relim(); ax_filt.autoscale_view(scaley=True, scalex=False)
        ax_psd.relim();  ax_psd.autoscale_view(scaley=True)
        overlay.set_text(metrics_overlay_text(f))
        return [*raw_lines, *filt_lines, *bars, overlay]

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
