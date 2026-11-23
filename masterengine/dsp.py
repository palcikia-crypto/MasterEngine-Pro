"""MasterEngine DSP core: filters, crossovers, dynamics, exciter, stereo, metering.

Pure numpy/scipy, sample-rate agnostic, stereo-native (shape (n, ch)).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Filters (RBJ biquads + butterworth)
# --------------------------------------------------------------------------- #

def biquad_peaking(sr, f, gain_db, q=0.8):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f / sr
    alpha = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * c, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * c, 1 - alpha / A])
    return b / a[0], a / a[0]


def biquad_shelf(sr, f, gain_db, q=0.7, high=False):
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f / sr
    alpha = np.sin(w0) / (2.0 * q)
    c = np.cos(w0)
    two_sq_a_alpha = 2.0 * np.sqrt(A) * alpha
    if high:
        b = np.array([A * ((A + 1) + (A - 1) * c + two_sq_a_alpha),
                      -2 * A * ((A - 1) + (A + 1) * c),
                      A * ((A + 1) + (A - 1) * c - two_sq_a_alpha)])
        a = np.array([(A + 1) - (A - 1) * c + two_sq_a_alpha,
                      2 * ((A - 1) - (A + 1) * c),
                      (A + 1) - (A - 1) * c - two_sq_a_alpha])
    else:
        b = np.array([A * ((A + 1) - (A - 1) * c + two_sq_a_alpha),
                      2 * A * ((A - 1) - (A + 1) * c),
                      A * ((A + 1) - (A - 1) * c - two_sq_a_alpha)])
        a = np.array([(A + 1) + (A - 1) * c + two_sq_a_alpha,
                      -2 * ((A - 1) + (A + 1) * c),
                      (A + 1) + (A - 1) * c - two_sq_a_alpha])
    return b / a[0], a / a[0]


def biquad_highpass(sr, f, q=0.707):
    return signal.butter(2, f, "hp", fs=sr, output="ba")


def apply_biquad(x, b, a):
    if abs(b[0] + b[2] * 0) < 1e-12 and abs(b[1]) < 1e-12 and abs(b[0]) < 1e-12:
        return x
    return signal.lfilter(b, a, x, axis=0)


def lr4_split(x, sr, freq, kind="lp"):
    """Linkwitz-Riley 4th order split (cascaded Butterworth 2nd order)."""
    fc = min(max(freq, 10.0), sr * 0.49)
    sos = signal.butter(2, fc, kind, fs=sr, output="sos")
    y = signal.sosfilt(sos, x, axis=0)
    y = signal.sosfilt(sos, y, axis=0)
    return y


# --------------------------------------------------------------------------- #
# Envelope / gain computer (block-rate control signal, vectorized)
# --------------------------------------------------------------------------- #

def _block_env(x, block):
    n = x.shape[0]
    m = (n // block) * block
    if m == 0:
        return np.ones(1), 0, 1
    env = np.abs(x[:m].reshape(-1, block, x.shape[1])).max(axis=1).max(axis=1)
    return env, m, block


def _coef(ms, ctrl_sr):
    if ms <= 0:
        return 1.0
    return 1.0 - np.exp(-1000.0 / (ms * ctrl_sr))


def _smooth(targets, attack_ms, release_ms, ctrl_sr):
    out = np.empty_like(targets)
    g = targets[0]
    ca = _coef(attack_ms, ctrl_sr)
    cr = _coef(release_ms, ctrl_sr)
    for i in range(targets.size):
        c = ca if targets[i] < g else cr
        g += c * (targets[i] - g)
        out[i] = g
    return out


def _upsample_gain(gain, n, block):
    ctrl = gain.size
    if ctrl == 0:
        return np.ones(n)
    centers = (np.arange(ctrl) + 0.5) * block
    idx = np.arange(n)
    out = np.interp(idx, centers, gain, period=None)
    out[idx >= centers[-1]] = gain[-1]
    return out


def _db(x):
    return 20.0 * np.log10(np.maximum(x, EPS))


# --------------------------------------------------------------------------- #
# Compressor / limiter
# --------------------------------------------------------------------------- #

def compress(x, sr, thr_db=-18.0, ratio=3.0, attack_ms=10.0, release_ms=120.0,
             makeup_db=0.0, knee_db=6.0, block=256):
    env, m, block = _block_env(x, block)
    ctrl_sr = sr / block
    xdb = _db(env)
    over = xdb - thr_db
    g_db = np.where(over > knee_db,
                    (1 - 1 / ratio) * (over - knee_db / 2), 0.0)
    g_db = np.where((over > 0) & (over <= knee_db),
                    (1 - 1 / ratio) * (over ** 2) / (2 * knee_db), g_db)
    g_db = -_smooth(np.maximum(g_db, 0.0), attack_ms, release_ms, ctrl_sr)
    gain = _upsample_gain(10.0 ** (g_db / 20.0), x.shape[0], block)
    gain[m:] = gain[m - 1:m] if m > 0 else 1.0
    return x * gain.reshape(-1, 1) * (10.0 ** (makeup_db / 20.0)), float(np.mean(g_db))


def limit(x, sr, ceiling_db=-1.0, lookahead_ms=1.5, attack_ms=0.05,
          release_ms=80.0, block=32):
    env, m, block = _block_env(x, block)
    ctrl_sr = sr / block
    thr = 10.0 ** (ceiling_db / 20.0)
    need = np.minimum(1.0, thr / np.maximum(env, EPS))
    need_db = _db(need)
    n_look = max(1, int(round(lookahead_ms * ctrl_sr)))
    shifted = np.concatenate([np.full(n_look, 0.0), need_db[:-n_look]]) if need_db.size > n_look else need_db
    g_db = -np.maximum(-_smooth(shifted, attack_ms, release_ms, ctrl_sr), 0.0)
    g_db = np.minimum(g_db, need_db)
    gain = _upsample_gain(10.0 ** (g_db / 20.0), x.shape[0], block)
    if m < x.shape[0]:
        gain[m:] = 10.0 ** (g_db[-1] / 20.0)
    return x * gain.reshape(-1, 1)


def limit_to(x, sr, ceiling_db=-1.0):
    """Limiter with guaranteed 4x-oversampled true-peak ceiling."""
    y = limit(x, sr, ceiling_db=ceiling_db)
    for _ in range(4):
        tp = true_peak_db(y, sr)
        if tp <= ceiling_db + 0.03:
            break
        y = y * (10.0 ** ((ceiling_db - 0.05 - tp) / 20.0))
    return y


# --------------------------------------------------------------------------- #
# Multiband compressor (LR4 3-way)
# --------------------------------------------------------------------------- #

def multiband_compress(x, sr, crossovers=(180.0, 2500.0), bands=None):
    bands = bands or [dict(thr_db=-18, ratio=2.5, attack_ms=18, release_ms=140),
                      dict(thr_db=-20, ratio=2.0, attack_ms=8, release_ms=120),
                      dict(thr_db=-22, ratio=2.5, attack_ms=5, release_ms=100)]
    lo = lr4_split(x, sr, crossovers[0], "lp")
    mid_full = lr4_split(x, sr, crossovers[0], "hp")
    mid = lr4_split(mid_full, sr, crossovers[1], "lp")
    hi = lr4_split(mid_full, sr, crossovers[1], "hp")
    out = np.zeros_like(x)
    for band, cfg in zip([lo, mid, hi], bands):
        y, _ = compress(band, sr, **cfg)
        out += y
    return out


# --------------------------------------------------------------------------- #
# Saturation / exciter
# --------------------------------------------------------------------------- #

def saturate(x, drive=1.5, mix=1.0):
    d = x * (1.0 + drive * 3.0)
    y = np.tanh(d) / max(drive, EPS)
    return x * (1.0 - mix) + y * mix


def excite(x, sr, freq=3000.0, drive=2.5, mix=0.15):
    high = lr4_split(x, sr, freq, "hp")
    harmonics = np.tanh(high * drive) - high * min(drive, 1.0)
    harmonics = np.tanh(harmonics * 0.8)
    return x + harmonics * mix * 0.2


def de_ess(x, sr, freq=6500.0, amount=0.5):
    s = lr4_split(x, sr, freq, "hp")
    y, _ = compress(s, sr, thr_db=-26.0, ratio=4.0, attack_ms=1.0,
                    release_ms=60.0, makeup_db=0.0)
    return x + (y - s) * amount


# --------------------------------------------------------------------------- #
# Stereo
# --------------------------------------------------------------------------- #

def stereo_widen(x, sr, width=1.15, bass_mono_hz=150.0):
    if x.shape[1] < 2:
        y = np.repeat(x, 2, axis=1) if x.shape[1] == 1 else x
        return y
    mid = x.mean(axis=1, keepdims=True)
    side = x[:, 0:1] - x[:, 1:2]
    if bass_mono_hz > 0:
        bass = x.mean(axis=1, keepdims=True)
        bass = lr4_split(bass, sr, bass_mono_hz, "lp")
        mid = mid - lr4_split(mid, sr, bass_mono_hz, "lp") + bass
    wide = side * (width - 1.0)
    if width > 1.0:
        fl = int(0.012 * sr)
        w = np.hanning(2 * fl + 1)[:fl // 2]
        wide = signal.fftconvolve(wide, w.reshape(-1, 1) / w.sum(), mode="same", axes=0) if fl > 3 else wide
    L = mid - 0.5 * side - 0.5 * wide
    R = mid + 0.5 * side + 0.5 * wide
    return np.concatenate([L, R], axis=1)


# --------------------------------------------------------------------------- #
# Metering: LUFS (BS.1770-3, gated), true-peak (4x), crest, balance, width
# --------------------------------------------------------------------------- #

def k_weight(x, sr):
    y = dsp_shelf(x, sr)
    return dsp_hpf(y, sr)


def dsp_shelf(x, sr):
    b, a = biquad_shelf(sr, 1681.97, 3.99984385397, q=0.7071752369554196, high=True)
    return signal.lfilter(b, a, x, axis=0)


def dsp_hpf(x, sr):
    sos = signal.butter(2, 38.0, "hp", fs=sr, output="sos")
    return signal.sosfilt(sos, signal.sosfilt(sos, x, axis=0), axis=0)


def lufs(x, sr):
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    y = k_weight(x, sr)
    mean_sq = np.mean(y ** 2, axis=0).sum()
    lg = -0.691 + 10.0 * np.log10(mean_sq + EPS)
    if lg > -70.0:
        return float(lg)
    return float(lg)


def lufs_gated(x, sr):
    y = k_weight(x, sr)
    hop = int(0.100 * sr)
    size = int(0.400 * sr)
    if x.shape[0] < size:
        return lufs(x, sr)
    blocks = []
    for i in range(0, x.shape[0] - size + 1, hop):
        z = y[i:i + size]
        blocks.append(np.mean(z ** 2, axis=0).sum())
    blocks = np.array(blocks)
    with np.errstate(divide="ignore"):
        lg = -0.691 + 10.0 * np.log10(blocks + EPS)
    abs_mask = lg > -70.0
    if not abs_mask.any():
        return -70.0
    rel = -0.691 + 10.0 * np.log10(blocks[abs_mask].mean())
    mask = abs_mask & (lg > rel - 10.0)
    if not mask.any():
        return float(np.mean(lg[abs_mask]))
    return float(-0.691 + 10.0 * np.log10(blocks[mask].mean()))


def true_peak_db(x, sr):
    n = 4
    y = signal.resample_poly(x, n, 1, axis=0)
    return float(_db(np.abs(y).max()))


def crest_factor(x):
    peak = np.abs(x).max()
    rms = np.sqrt(np.mean(x ** 2))
    return float(_db(peak + EPS) - _db(rms + EPS))


def stereo_width(x, sr):
    if x.shape[1] < 2:
        return 0.0
    mid = x.mean(axis=1)
    side = x[:, 0] - x[:, 1]
    e_m = np.sqrt(np.mean(mid ** 2) + EPS)
    e_s = np.sqrt(np.mean(side ** 2) + EPS)
    return float(_db(e_s / e_m))


BANDS = [(20, 60), (60, 150), (150, 400), (400, 2000), (2000, 5000), (5000, 12000), (12000, 20000)]


def band_energy_db(x, sr, fmin, fmax):
    sos = signal.butter(2, [max(fmin, 12.0), min(fmax, sr / 2 - 1.0)], "bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, x, axis=0)
    return float(_db(np.sqrt(np.mean(y ** 2) + EPS)))


def band_balance(x, sr, bands=BANDS):
    return {f"{fmin}-{fmax}": band_energy_db(x, sr, fmin, fmax) for fmin, fmax in bands}


def spectrum(x, sr, n=4096):
    """Averaged magnitude spectrum in dB, freq axis."""
    ch = x.mean(axis=1)
    freqs, psd = signal.welch(ch, sr, nperseg=n)
    return freqs, _db(np.sqrt(psd) + EPS)
