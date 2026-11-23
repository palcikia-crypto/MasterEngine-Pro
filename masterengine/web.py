"""FastAPI backend for the MasterEngine web panel."""
from __future__ import annotations

import json
import os
import tempfile
import uuid

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import analysis, master, stems

DATA_DIR = os.environ.get("ME_DATA_DIR", os.path.join(tempfile.gettempdir(), "masterengine_data"))
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(title="MasterEngine", version="1.0.0")
STE = {}


def _job_dir(jid):
    return os.path.join(DATA_DIR, jid)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), use_stems: bool = Form(False)):
    jid = uuid.uuid4().hex[:12]
    d = _job_dir(jid)
    os.makedirs(d, exist_ok=True)
    data, sr = sf.read(file.file, always_2d=True, dtype="float32")
    sf.write(os.path.join(d, "original.wav"), data, sr, subtype="PCM_16")
    x = data.astype(np.float64)[:, :2]
    if x.shape[1] == 1 or x.shape[0] > 0 and x.shape[1] < 2:
        x = np.repeat(x, 2, axis=1)
    STE[jid] = {"x": x, "sr": sr, "stems_cache": None, "use_stems": use_stems}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump({"name": file.filename, "sr": sr, "use_stems": use_stems}, f)
    return {"id": jid, "name": file.filename, "sr": sr,
            "seconds": round(float(x.shape[0]) / sr, 2),
            "before": master.public_feats(analysis.analyze(x, sr))}


class ProcessRequest(BaseModel):
    params: dict | None = None
    stems: bool = False
    refine_iters: int = 4


@app.post("/api/master/{jid}")
def do_master(jid: str, req: ProcessRequest):
    if jid not in STE:
        return JSONResponse({"error": "unknown job"}, status_code=404)
    st = STE[jid]
    x, sr = st["x"], st["sr"]
    d = _job_dir(jid)

    res, y = master.process(x, sr, params=req.params, refine_iters=req.refine_iters)
    mastered_path = os.path.join(d, "mastered.wav")
    sf.write(mastered_path, y.astype(np.float32), sr, subtype="PCM_16")

    res["after"] = master.public_feats(analysis.analyze(y, sr))
    res["files"] = {"original": f"/api/audio/j/{jid}/original.wav",
                    "mastered": f"/api/audio/j/{jid}/mastered.wav"}

    if req.stems:
        if st["stems_cache"] is None:
            raw = stems.separate(x, sr)
            st["stems_cache"] = {k: v.astype(np.float32) for k, v in raw.items()}
        stem_res = {}
        mix = np.zeros_like(x)
        for name, s in st["stems_cache"].items():
            f_in = analysis.analyze(s.astype(np.float64), sr)
            p = stems.build_stem_params(name, f_in)
            ys = master.apply_chain(s.astype(np.float64), sr, p)
            ft = analysis.analyze(ys, sr)
            for _ in range(2):
                p = master.MODEL.refine_step(p, ft)
                ys = master.safe_render(s.astype(np.float64), sr, p)
                ft = analysis.analyze(ys, sr)
            sf.write(os.path.join(d, f"stem_{name}.wav"), s, sr, subtype="PCM_16")
            sf.write(os.path.join(d, f"stemm_{name}.wav"), ys.astype(np.float32), sr, subtype="PCM_16")
            stem_res[name] = {"before": master.public_feats(f_in), "after": master.public_feats(ft),
                              "params": p,
                              "files": {"original": f"/api/audio/j/{jid}/stem_{name}.wav",
                                        "mastered": f"/api/audio/j/{jid}/stemm_{name}.wav"}}
            mix += ys
        sf.write(os.path.join(d, "stem_mix.wav"), mix.astype(np.float32), sr, subtype="PCM_16")
        res["stems"] = stem_res
        res["files"]["stem_mix"] = f"/api/audio/j/{jid}/stem_mix.wav"

    with open(os.path.join(d, "last_result.json"), "w") as f:
        json.dump(res, f, default=float)
    return res


@app.get("/api/audio/j/{jid}/{fname}")
def audio(jid: str, fname: str):
    p = os.path.join(_job_dir(jid), fname)
    if not os.path.isfile(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="audio/wav", filename=fname)


def _peaks(path, n=800):
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    m = (data.shape[0] // n) * n
    if m == 0:
        return np.zeros(n).tolist()
    blk = data[:m].reshape(n, -1, data.shape[1])
    return blk.max(axis=1).max(axis=1).tolist()


@app.get("/api/peaks/{jid}/{fname}")
def peaks(jid: str, fname: str, n: int = 800):
    p = os.path.join(_job_dir(jid), fname)
    if not os.path.isfile(p):
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"n": n, "peaks": _peaks(p, n)}


@app.get("/api/health")
def health():
    return {"ok": True, "engine": "MasterEngine 1.0"}


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
