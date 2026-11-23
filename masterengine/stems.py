"""Stem separation (chunked HPSS + optional Demucs) and per-stem mastering.

Default engine: frame-grid STFT harmonic/percussive separation (Fitzgerald
median masking) processed in time-chunks with crossfaded overlap-add so memory
stays bounded for arbitrarily long tracks. If `demucs` is installed
(`pip install demucs torch torchaudio`), it is used automatically instead.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from . import analysis, master

N_FFT = 4096
HOP = 1024
CHUNK = 262144
OVERLAP = 32768

STEM_ORDER = ["drums", "bass", "vocals", "other"]


def demucs_stems(audio, sr, model_name="htdemucs"):
    import torch
    audio_t = torch.as_tensor(audio.T).float()
    if audio.shape[1] == 1:
        audio_t = audio_t.repeat(1, 2, 1)
    audio_t = audio_t.unsqueeze(0)
    model = __import__("demucs.pretrained", fromlist=["get_model"]).get_model(model_name)
    wav = model(waveforms=audio_t, sample_rate=int(sr))
    out = wav[0].numpy().T
    names = list(model.sources)
    return {name: out[:, :, i].copy() if out.ndim == 3 else out[:, i, :].T for i, name in enumerate(names)}


def _soft_band_mask_f(freqs, lo, hi, soft=0.35):
    lo = max(lo, 1.0)
    m = 1.0 / (1.0 + np.exp(np.clip((lo - freqs) / (lo * soft + 1.0), -60, 60)))
    m *= 1.0 / (1.0 + np.exp(np.clip((freqs - hi) / (hi * soft + 1.0), -60, 60)))
    return m


def _chunk_loop(x, sr, chunk_cb, progress_cb=None):
    """Run chunk_cb(seg, freqs, Z) per time-chunk with crossfaded overlap-add."""
    n = x.shape[0]
    ramp = OVERLAP // 2
    t_i = np.linspace(0, np.pi, max(ramp, 1))
    win = np.sqrt(0.5 - 0.5 * np.cos(t_i))
    weights = np.zeros(n)
    outs = {}
    pos = 0
    step = 0
    n_steps = max(1, math_ceil(n / CHUNK))
    while pos < n:
        end = min(pos + CHUNK, n)
        lo = max(0, pos - OVERLAP)
        hi = min(n, end + OVERLAP)
        seg_len = hi - lo
        env = np.ones(seg_len)
        if lo > 0:
            env[:ramp] *= win
        if hi < n:
            env[-ramp:] *= win
        seg = x[lo:hi].astype(np.float64)
        freqs, _, Z = signal.stft(seg.T, axis=1, nperseg=N_FFT, noverlap=N_FFT - HOP,
                                  window="hann", padded=True)
        stem_frames = chunk_cb(freqs, Z, seg_len)
        for name, frames in stem_frames.items():
            buf = outs.setdefault(name, np.zeros((n, x.shape[1]), dtype=np.float32))
            _, y = signal.istft(frames, nperseg=N_FFT, noverlap=N_FFT - HOP,
                                window="hann", time_axis=-1, freq_axis=-2)
            y = y.T[:seg_len]
            buf[lo:hi] = (buf[lo:hi].astype(np.float64) + y * env[:, None]).astype(np.float32)
        weights[lo:hi] += env
        pos = end
        step += 1
        if progress_cb:
            progress_cb(step, n_steps)
    for name, buf in outs.items():
        buf /= np.maximum(weights, 1e-6)[:, None]
    for name in STEM_ORDER:
        outs.setdefault(name, np.zeros((n, x.shape[1]), dtype=np.float32))
    return outs


def math_ceil(x):
    import math
    return math.ceil(x)


def hpss_masks(freqs, Z):
    mag = np.abs(Z).mean(axis=0)
    harm = median_filter(mag, size=(1, 17))
    perc = median_filter(mag, size=(17, 1))
    h = harm ** 2 / (harm ** 2 + perc ** 2 + 1e-12)
    mids = _soft_band_mask_f(freqs, 220.0, 8000.0)[:, None]
    lows = _soft_band_mask_f(freqs, 20.0, 190.0)[:, None]
    drums = (1 - h)[None]
    vocals = (h * mids)[None]
    bass = (h * lows * np.clip(1 - mids, 0, 1))[None]
    other = np.clip(1.0 - drums - vocals - bass, 0, 1)
    return {"drums": drums, "vocals": vocals, "bass": bass, "other": other}


def separate(x, sr, engine="auto", progress_cb=None) -> dict:
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if engine in ("auto", "demucs"):
        try:
            return demucs_stems(x, sr)
        except ImportError:
            if engine == "demucs":
                raise
        except Exception:
            if engine == "demucs":
                raise

    def cb(freqs, Z, _seg_len):
        return {name: Z * mask for name, mask in hpss_masks(freqs, Z).items()}

    outs = _chunk_loop(x, sr, cb, progress_cb)
    return {k: v.astype(np.float64) for k, v in outs.items()}


STEM_REFERENCE = {
    "drums": {"comp_thr": -14, "comp_ratio": 2.0, "saturation": 0.25, "exciter_mix": 0.2,
              "balance": {"20-60": -7.5, "60-150": -3.0, "150-400": -5.0, "400-2000": -3.0,
                          "2000-5000": -5.0, "5000-12000": -8.0, "12000-20000": -16.0}},
    "bass": {"comp_thr": -16, "comp_ratio": 2.5, "saturation": 0.2, "exciter_mix": 0.06,
             "balance": {"20-60": -6.0, "60-150": -2.0, "150-400": -5.0}},
    "vocals": {"comp_thr": -18, "comp_ratio": 2.2, "saturation": 0.12, "exciter_mix": 0.16,
               "balance": {"400-2000": -3.0, "2000-5000": -5.0, "5000-12000": -9.0}},
    "other": {"comp_thr": -20, "comp_ratio": 1.8, "saturation": 0.1, "exciter_mix": 0.1,
              "balance": {"2000-5000": -5.0, "5000-12000": -8.0}},
}


def build_stem_params(name, feats, width=1.0):
    from . import dsp
    p = {
        "hpf_hz": {"drums": 28.0, "bass": 20.0, "vocals": 90.0, "other": 120.0}.get(name, 40.0),
        "band_gains_db": {},
        "xover_hz": [220.0, 4500.0],
        "comp": [],
        "saturation": 0.15,
        "exciter": {"freq": 3000.0, "mix": 0.1},
        "de_ess": {"freq": 7000.0, "amount": 0.25 if name == "vocals" else 0.0},
        "stereo_width": width,
        "bass_mono_hz": 120.0 if name != "bass" else 0.0,
        "target_lufs": -14.0,
        "ceiling_dbtp": -1.0,
    }
    ref = STEM_REFERENCE.get(name, {})
    order = [f"{a}-{b}" for a, b in dsp.BANDS]
    base = {k: 0.0 for k in order}
    keys = sorted(feats["balance_db"].keys(), key=lambda k: float(k.split("-")[0]))
    cur = np.array([feats["balance_db"][k] for k in keys])
    cur = cur - cur.max()
    for k in keys:
        if k in ref and k in base:
            i = keys.index(k)
            base[k] = float(np.clip((cur[i] - ref[k]) * -0.4, -6, 6))
    p["band_gains_db"] = base
    cthr = ref.get("comp_thr", -18.0)
    crat = ref.get("comp_ratio", 2.0)
    p["comp"] = [dict(thr_db=cthr, ratio=crat, attack_ms=15.0, release_ms=140.0)] * 3
    p["saturation"] = ref.get("saturation", 0.15)
    p["exciter"]["mix"] = ref.get("exciter_mix", 0.1)
    return p


def master_stems(x, sr, refine_iters=2, progress_cb=None) -> dict:
    stems = separate(x, sr, progress_cb=progress_cb)
    out = {}
    feats_in = {}
    feats_out = {}
    for name, s in stems.items():
        if s.shape[0] < 64:
            s = np.pad(s, ((0, 64 - s.shape[0]), (0, 0)))
        f_in = analysis.analyze(s, sr)
        pre_db = float(np.clip(-22.0 - f_in["lufs"], -20.0, 20.0))
        sw = s * (10.0 ** (pre_db / 20.0))
        p = build_stem_params(name, f_in)
        y = master.apply_chain(sw, sr, p)
        ft = analysis.analyze(y, sr)
        for _ in range(refine_iters):
            p = master.MODEL.refine_step(p, ft)
            y = master.safe_render(sw, sr, p)
            ft = analysis.analyze(y, sr)
        out[name] = (s, y, p)
        feats_in[name] = f_in
        feats_out[name] = ft
    return {"stems": out, "mix": sum(y for _, y, _ in out.values()), "before": feats_in, "after": feats_out}
