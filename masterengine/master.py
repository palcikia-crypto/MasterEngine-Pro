"""Mastering chain renderer + closed-loop orchestrator."""
from __future__ import annotations

import numpy as np

from . import analysis, dsp, ml

MODEL = ml.SettingsModel()


def apply_chain(x, sr, p) -> np.ndarray:
    y = x if x.ndim == 2 else np.stack([x, x], axis=1)

    y = dsp.apply_biquad(y, *dsp.biquad_highpass(sr, p["hpf_hz"]))

    keys = sorted(p["band_gains_db"].keys(), key=lambda k: float(k.split("-")[0]))
    for k in keys:
        fmin, fmax = (float(v) for v in k.split("-"))
        g = p["band_gains_db"][k]
        if abs(g) < 0.05:
            continue
        if fmin <= 60:
            f, q = np.sqrt(fmin * fmax) * 1.2, 0.9
            y = dsp.apply_biquad(y, *dsp.biquad_shelf(sr, f, g, q=q, high=False))
        elif fmax >= 12000:
            f, q = 10000.0, 0.8
            y = dsp.apply_biquad(y, *dsp.biquad_shelf(sr, f, g, q=q, high=True))
        else:
            f = np.sqrt(fmin * fmax)
            q = 1.1 if fmax < 300 else 0.9
            y = dsp.apply_biquad(y, *dsp.biquad_peaking(sr, f, g, q=q))

    y = dsp.multiband_compress(y, sr, tuple(p["xover_hz"]), p["comp"])

    if p["saturation"] > 0.01:
        y = dsp.saturate(y, drive=p["saturation"] * 3, mix=0.5)
    if p["exciter"]["mix"] > 0.005:
        y = dsp.excite(y, sr, freq=p["exciter"]["freq"], mix=p["exciter"]["mix"])
    if p["de_ess"]["amount"] > 0.005:
        y = dsp.de_ess(y, sr, freq=p["de_ess"]["freq"], amount=p["de_ess"]["amount"])

    y = dsp.stereo_widen(y, sr, width=p["stereo_width"], bass_mono_hz=p["bass_mono_hz"])

    y = y * (10.0 ** (p.get("_output_gain_db", 0.0) / 20.0))
    y = dsp.limit_to(y, sr, ceiling_db=p["ceiling_dbtp"])
    return y


def safe_render(x, sr, p, max_rms_blowup=12.0) -> np.ndarray:
    y = apply_chain(x, sr, p)
    ref = np.sqrt(np.mean(x ** 2)) + 1e-12
    cur = np.sqrt(np.mean(y ** 2))
    if cur > ref * max_rms_blowup or not np.all(np.isfinite(y)):
        p2 = dict(p)
        p2["_output_gain_db"] = p2.get("_output_gain_db", 0.0) - 12.0
        y = apply_chain(x, sr, p2)
    return np.nan_to_num(y)


def loss(feats_out, p) -> float:
    from . import dsp
    keys = _keys()
    cur = analysis.balance_vector(feats_out)
    order = keys
    err = 0.0
    ref_full = [ml.REFERENCE_BALANCE.get(k, -12.0) for k in order]
    for i, k in enumerate(order):
        w = 0.5 if ml.REFERENCE_BALANCE.get(k, -12.0) < -17.5 else 1.0
        err += w * (cur[i] - ref_full[i]) ** 2
    d_lufs = (feats_out["lufs"] - p["target_lufs"]) ** 2 * 1.5
    crest = feats_out["crest_db"]
    d_crest = (max(0, ml.TARGET_CREST_RANGE[0] - crest) + max(0, crest - ml.TARGET_CREST_RANGE[1] - 1.0)) ** 2
    return err * 0.05 + d_lufs + d_crest * 0.5


def _keys():
    from . import dsp as _d
    return [f"{a}-{b}" for a, b in _d.BANDS]


def process(x, sr, params=None, refine_iters=7, save_experience=True) -> dict:
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    feats_in = analysis.analyze(x, sr)
    if params is None:
        p = MODEL.predict(feats_in, sr)
    else:
        p = MODEL.sanitize(dict(params))
        p["target_lufs"] = params.get("target_lufs", ml.TARGET_LUFS)
        p["_user"] = True

    pre_db = float(np.clip(-18.0 - feats_in["lufs"], -20.0, 20.0))
    x = x * (10.0 ** (pre_db / 20.0))

    best = None
    for it in range(refine_iters):
        y = safe_render(x, sr, p)
        ft = analysis.analyze(y, sr)
        L = loss(ft, p)
        if best is None or L < best[0]:
            best = (L, y.copy(), ft, json_copy(p))
        if it < refine_iters - 1:
            p = MODEL.refine_step(json_copy(p), ft)
    L, y, feats_out, p_used = best

    result = {
        "preset_params": MODEL.predict(feats_in, sr),
        "params": p_used,
        "loss": L,
        "before": public_feats(feats_in),
        "after": public_feats(feats_out),
        "pre_gain_db": pre_db,
    }
    if save_experience:
        MODEL.save_experience(feats_in, {k: v for k, v in p_used.items() if k != "_output_gain_db"})
    return result, y


def json_copy(d):
    import json
    return json.loads(json.dumps(d))


def public_feats(f):
    return {k: v for k, v in f.items() if k != "peak_lin"}
