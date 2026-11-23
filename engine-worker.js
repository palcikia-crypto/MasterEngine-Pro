/* MasterEngine worker — pure JS engine, everything off the main thread. v2 layer included. */
importScripts("js-engine.js");
importScripts("engine-v2.js");
const E = self.MasterEngine;
const V2 = self.MasterEngineV2;

const post = (m, tr) => self.postMessage(m, tr || []);
const say = (msg, pct) => post({ t: "status", msg, pct: pct == null ? null : pct });

self.onmessage = async (e) => {
  const d = e.data;
  if (d.t !== "load") return;
  try {
    const t0 = Date.now();
    const f32 = new Float32Array(d.buf);
    const n = d.n | 0;
    const x = E.planar32(n);
    x.L.set(f32.subarray(0, n));
    x.R.set(f32.subarray(n, 2 * n));
    say("Трек принят: " + n + " семплов, " + d.sr + " Hz");
    const main = await V2.processFullV2(
      x, d.sr,
      { profile: d.profile || "auto", intensity: d.intensity, params: d.params || null, stems: !!d.stems, iters: d.iters },
      (msg, pct) => say(msg, pct == null ? null : pct)
    );
    const files = {};
    const res = main.result;
    if (d.stems) {
      const sm = main.sm;
      for (const name of Object.keys(sm.stems)) {
        const st = sm.stems[name];
        files["/session/stem_" + name + ".wav"] = E.encodeWav(st.raw, d.sr).buffer;
        files["/session/stemm_" + name + ".wav"] = E.encodeWav(st.master, d.sr).buffer;
      }
      files["/session/stem_mix.wav"] = E.encodeWav(sm.mix, d.sr).buffer;
    }
    files["/session/mastered.wav"] = E.encodeWav(main.y, d.sr).buffer;
    files["/session/original.wav"] = E.encodeWav(x, d.sr).buffer;
    res.frames = Object.keys(files);
    say("Готово за " + ((Date.now() - t0) / 1000).toFixed(1) + " с.", null);
    post({ t: "result", res, files }, Object.values(files));
  } catch (err) {
    post({ t: "error", msg: String((err && err.stack) || err).split("\n").slice(-4).join(" | ") });
  }
};

post({ t: "ready" });
