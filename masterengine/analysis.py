"""Feature extraction for the ML auto-mastering layer."""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import dsp


def analyze(x, sr) -> dict:
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    peak = np.abs(x).max()
    feats = {
        "lufs": dsp.lufs_gated(x, sr),
        "peak_dbtp": dsp.true_peak_db(x, sr),
        "crest_db": dsp.crest_factor(x),
        "stereo_width_db": dsp.stereo_width(x, sr),
        "balance_db": dsp.band_balance(x, sr),
        "peak_lin": float(peak),
    }
    freqs, spec = dsp.spectrum(x, sr)
    m = (freqs > 40) & (freqs < 16000)
    if m.any():
        feats["centroid_hz"] = float(np.sum(freqs[m] * (10 ** (spec[m] / 20))) /
                                     (np.sum(10 ** (spec[m] / 20)) + 1e-12))
    else:
        feats["centroid_hz"] = 1000.0
    return feats


def balance_vector(feats) -> np.ndarray:
    keys = sorted(feats["balance_db"].keys(), key=lambda k: float(k.split("-")[0]))
    v = np.array([feats["balance_db"][k] for k in keys])
    return v - v.max()


def describe(feats) -> str:
    return (f"LUFS {feats['lufs']:.1f} | TP {feats['peak_dbtp']:.1f} dBTP | "
            f"crest {feats['crest_db']:.1f} dB | width {feats['stereo_width_db']:.1f} dB | "
            f"centroid {feats.get('centroid_hz', 0):.0f} Hz")
