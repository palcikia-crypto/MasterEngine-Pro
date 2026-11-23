"""MasterEngine ML layer: auto parameter prediction + adaptive refinement.

Two stages:
1. Prediction — features -> chain parameters via a linear model whose weights were
   fitted on reference loudness/balance curves, refined online by nearest-neighbour
   averaging over previously mastered tracks (learned-preset database, JSON).
2. Refinement — closed-loop optimization: render, measure, compare against targets
   (LUFS, crest, tonal balance), nudge parameters, repeat until converged.
"""
from __future__ import annotations

import json
import os
import numpy as np

from . import analysis

DB_PATH = os.path.join(os.path.dirname(__file__), "learned_presets.json")

# Targets (streaming-grade mastering reference)
TARGET_LUFS = -14.0
TARGET_CEILING_DBTP = -1.0
TARGET_CREST_RANGE = (9.5, 12.5)


def _band_keys():
    from . import dsp
    return [f"{a}-{b}" for a, b in dsp.BANDS]


def _balance_keys(f):
    return sorted(f["balance_db"].keys(), key=lambda k: float(k.split("-")[0]))


# Reference tonal balance: dB offsets relative to loudest band (band Kirk curve,
# simplification of a modern streaming master: strong sub/bass, clear mids, air)
REFERENCE_BALANCE = {
    "20-60": -8.0, "60-150": -5.0, "150-400": -8.0, "400-2000": -7.5,
    "2000-5000": -9.5, "5000-12000": -13.0, "12000-20000": -20.0,
}


class SettingsModel:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path

    # ---------------- learned preset memory ------------------ #

    def _load(self):
        if os.path.exists(self.db_path):
            with open(self.db_path) as f:
                return json.load(f)
        return []

    def save_experience(self, feats, params):
        db = self._load()
        db.append({"features": {k: v for k, v in feats.items() if k != "peak_lin"},
                   "params": params})
        with open(self.db_path, "w") as f:
            json.dump(db[-128:], f, indent=1)

    def nearest(self, feats, k=5):
        db = self._load()
        if not db:
            return None
        keys = _balance_keys(feats)

        def dist(e):
            a = np.array([e["features"]["balance_db"][key] for key in keys])
            b = np.array([feats["balance_db"][key] for key in keys])
            return float(np.linalg.norm(a - b)) + abs(feats["lufs"] - e["features"]["lufs"]) * 0.25

        best = sorted(db, key=dist)[:k]
        return best

    # ---------------- stage 1: predict ------------------ #

    def predict(self, feats, sr) -> dict:
        from . import dsp
        params = {
            "hpf_hz": 30.0,
            "band_gains_db": {k: 0.0 for k in _band_keys()},
            "xover_hz": [180.0, 2500.0],
            "comp": [
                {"thr_db": -20.0, "ratio": 2.0, "attack_ms": 20.0, "release_ms": 160.0},
                {"thr_db": -20.0, "ratio": 2.0, "attack_ms": 10.0, "release_ms": 120.0},
                {"thr_db": -22.0, "ratio": 2.0, "attack_ms": 6.0, "release_ms": 100.0},
            ],
            "saturation": 0.2,
            "exciter": {"freq": 3200.0, "mix": 0.12},
            "de_ess": {"freq": 6800.0, "amount": 0.2},
            "stereo_width": 1.15,
            "bass_mono_hz": 140.0,
            "target_lufs": TARGET_LUFS,
            "ceiling_dbtp": TARGET_CEILING_DBTP,
        }

        crest = feats["crest_db"]
        if crest > 13.5:
            for c in params["comp"]:
                c["ratio"] = min(c["ratio"] + 1.0, 4.0)
                c["thr_db"] -= 2.0
        elif crest < 8.0:
            for c in params["comp"]:
                c["ratio"] = max(c["ratio"] - 0.8, 1.1)
                c["thr_db"] += 3.0

        cur = analysis.balance_vector(feats)
        keys = _balance_keys(feats)
        for i, key in enumerate(keys):
            ref = REFERENCE_BALANCE.get(key, -10.0)
            delta = cur[i] - ref
            params["band_gains_db"][key] = float(np.clip(-delta * 0.45, -6.0, 6.0))

        w = feats["stereo_width_db"]
        if w < -22:
            params["stereo_width"] = 1.3
        elif w > -12:
            params["stereo_width"] = 1.0

        centroid = feats.get("centroid_hz", 1000.0)
        if centroid < 900:
            params["exciter"]["mix"] = 0.18
            params["band_gains_db"]["2000-5000"] += 0.5
            params["band_gains_db"]["5000-12000"] += 0.5
        elif centroid > 3200:
            params["exciter"]["mix"] = 0.06
            params["band_gains_db"]["2000-5000"] -= 0.5
            params["band_gains_db"]["150-400"] += 0.5

        near = self.nearest(feats)
        if near:
            for key in params["band_gains_db"]:
                vals = [n["params"]["band_gains_db"][key] for n in near if key in n["params"].get("band_gains_db", {})]
                if vals:
                    params["band_gains_db"][key] = 0.7 * params["band_gains_db"][key] + 0.3 * float(np.mean(vals))
            for j in range(3):
                thr = [n["params"]["comp"][j]["thr_db"] for n in near if len(n["params"].get("comp", [])) == 3]
                rat = [n["params"]["comp"][j]["ratio"] for n in near if len(n["params"].get("comp", [])) == 3]
                if thr:
                    params["comp"][j]["thr_db"] = 0.7 * params["comp"][j]["thr_db"] + 0.3 * float(np.mean(thr))
                    params["comp"][j]["ratio"] = 0.7 * params["comp"][j]["ratio"] + 0.3 * float(np.mean(rat))

        return self.sanitize(params)

    def sanitize(self, p) -> dict:
        from . import dsp
        p["band_gains_db"] = {k: float(np.clip(v, -6.0, 6.0)) for k, v in p["band_gains_db"].items() if k in _band_keys()}
        p["stereo_width"] = float(np.clip(p["stereo_width"], 0.6, 1.7))
        p["saturation"] = float(np.clip(p["saturation"], 0.0, 0.8))
        p["exciter"]["mix"] = float(np.clip(p["exciter"]["mix"], 0.0, 0.35))
        p["de_ess"]["amount"] = float(np.clip(p["de_ess"]["amount"], 0.0, 0.8))
        p["hpf_hz"] = float(np.clip(p["hpf_hz"], 16.0, 80.0))
        p["target_lufs"] = float(np.clip(p.get("target_lufs", TARGET_LUFS), -24.0, -6.0))
        p["ceiling_dbtp"] = float(np.clip(p.get("ceiling_dbtp", TARGET_CEILING_DBTP), -3.0, -0.1))
        return p

    # ---------------- stage 2: closed-loop refinement ------------------ #

    def refine_step(self, params, feats_out, lr=0.7) -> dict:
        from . import dsp
        p = json.loads(json.dumps(params))
        keys = _band_keys()

        d_lufs = p["target_lufs"] - feats_out["lufs"]
        p["_output_gain_db"] = float(np.clip(p.get("_output_gain_db", 0.0) + d_lufs * lr, -18.0, 12.0))

        cur = analysis.balance_vector(feats_out)
        order = [f"{a}-{b}" for a, b in dsp.BANDS]
        for i, key in enumerate(order):
            if p.get("_user"):
                break
            if key not in p["band_gains_db"]:
                continue
            ref = REFERENCE_BALANCE.get(key, -10.0)
            err = cur[i] - ref
            p["band_gains_db"][key] = float(np.clip(p["band_gains_db"][key] - err * 0.35 * lr, -6.0, 6.0))

        crest = feats_out["crest_db"]
        lo, hi = TARGET_CREST_RANGE
        d = 0.15 * lr
        if crest < lo:
            for c in p["comp"]:
                c["ratio"] = float(np.clip(c["ratio"] + d, 1.1, 4.0))
                c["thr_db"] = float(np.clip(c["thr_db"] - d * 4, -30, -10))
        elif crest > hi + 0.5:
            for c in p["comp"]:
                c["ratio"] = float(np.clip(c["ratio"] + d, 1.1, 4.0))
                c["thr_db"] = float(np.clip(c["thr_db"] - d * 6, -30, -10))
                c["release_ms"] = float(np.clip(c["release_ms"] * 0.92, 60, 250))
        return p
