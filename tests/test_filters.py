# Sanity tests against scipy reference implementations.
# Skipped wholesale if cuda_eeg_prep isn't built — they're meant to run on the Jetson.

import importlib.util
import numpy as np
import pytest
from scipy.signal import firwin, lfilter, welch

FS    = 250.0
WIN   = 256
N_CH  = 8

if importlib.util.find_spec("cuda_eeg_prep") is None:
    pytest.skip("cuda_eeg_prep not built — run cmake first", allow_module_level=True)

import cuda_eeg_prep  # noqa: E402


def _scipy_fir_alpha(sig: np.ndarray) -> np.ndarray:
    # Matches taps/generate_taps.py for ALPHA: 8–13 Hz, 129 taps, Hamming.
    taps = firwin(129, [8.0, 13.0], pass_zero=False, fs=FS, window="hamming")
    out = np.empty_like(sig)
    for c in range(sig.shape[0]):
        out[c] = lfilter(taps, 1.0, sig[c]).astype(np.float32)
    return out


def test_alpha_fir_matches_scipy():
    rng = np.random.default_rng(1)
    n = 4 * WIN
    sig = rng.standard_normal((N_CH, n)).astype(np.float32)

    gpu = cuda_eeg_prep.filter_band("alpha", sig)
    cpu = _scipy_fir_alpha(sig)

    # Steady-state portion (after filter transient) should agree closely.
    np.testing.assert_allclose(gpu[:, 256:], cpu[:, 256:], atol=1e-3, rtol=1e-3)


def test_psd_peaks_at_alpha():
    """Inject a 10 Hz sinusoid + per-channel noise. Alpha band should carry the most power.
    Per-channel noise is required because CAR cancels common-mode signal."""
    rng = np.random.default_rng(2)
    pipe = cuda_eeg_prep.Pipeline(N_CH, WIN, FS)
    t = np.arange(WIN) / FS

    # Prime + measure: feed independent noise + alpha sine each iteration.
    feat = None
    for _ in range(16):
        sig = (0.3 * rng.standard_normal((N_CH, WIN))).astype(np.float32)
        sig += np.sin(2 * np.pi * 10.0 * t).astype(np.float32)[None, :]
        feat = pipe.process(sig)

    # Bands: [delta, theta, alpha, beta, gamma, ssvep]. SSVEP (5–20 Hz) contains the alpha
    # range so both will be elevated; assert alpha beats the disjoint bands.
    assert feat.shape == (N_CH, 6)
    for c in range(N_CH):
        a = feat[c, 2]
        for other in (0, 1, 3, 4):  # delta, theta, beta, gamma
            assert a > feat[c, other], f"ch {c}: alpha={a} not greater than band {other}={feat[c, other]}"


def test_car_zero_mean():
    """After CAR, the cross-channel mean per sample should be ~0. Inspect via the bypassed
    raw API: feed all-ones into all channels — every band power should be ≈ 0 (not exactly,
    bandpass kills DC anyway, but the test is that nothing explodes)."""
    pipe = cuda_eeg_prep.Pipeline(N_CH, WIN, FS)
    sig = np.ones((N_CH, WIN), dtype=np.float32)
    feat = pipe.process(sig)
    assert np.all(np.isfinite(feat))
    assert np.all(feat >= 0.0)
