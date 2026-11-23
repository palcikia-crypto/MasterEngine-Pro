/* MasterEngine v2 — profile-based enhancement layer on top of js-engine.js.
 * Additive: does NOT modify the original engine. Loads after js-engine.js.
 * Exports MasterEngineV2 with processAsyncV2 / masterStemsAsyncV2 / processFullV2 / enhanceV2.
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.MasterEngineV2 = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const E = (typeof self !== "undefined" && self.MasterEngine) ||
    (typeof require !== "undefined" && typeof require === "function" ? require("./js-engine.js") : null);
  if (!E) throw new Error("MasterEngineV2: js-engine.js must be loaded first (module or self.MasterEngine)");

  const EPS = 1e-12;
  const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const db2lin = d => Math.pow(10, d / 20);

  // ------------------------------------------------------------ profiles --
  const PROFILES = {
    auto: {
      name: "auto", label: "Авто (универсально)",
      deglass: 0.5, glassHz: [2300, 3600], glassCutDb: 2.0,
      warm: 0.15, warmEven: 0.20, airDb: 1.5, airHz: 11000,
      punch: 0.30, width: 1.10, widthLowKeep: 0.85,
      gains: {}, comp: null, exciterMax: 0.18, satMax: 0.45, deEssMin: null, lufs: null,
    },
    suno: {
      name: "suno", label: "Suno — живой вокал",
      deglass: 1.0, glassHz: [2200, 3600], glassCutDb: 3.2,
      warm: 0.45, warmEven: 0.30, airDb: 2.0, airHz: 10500,
      punch: 0.40, width: 1.18, widthLowKeep: 0.70,
      gains: { "2000-5000": -1.0, "150-400": 0.5 }, comp: null,
      exciterMax: 0.07, satMax: 0.50, deEssMin: 0.22, lufs: -13.5,
    },
    warm: {
      name: "warm", label: "Тёплый / винтаж",
      deglass: 0.7, glassHz: [2000, 3400], glassCutDb: 2.5,
      warm: 0.60, warmEven: 0.35, airDb: 0.5, airHz: 10500,
      punch: 0.25, width: 1.05, widthLowKeep: 0.75,
      gains: { "150-400": 1.0, "2000-5000": -0.5 }, comp: [{ thr_db: -19, ratio: 2.4 }, { thr_db: -19, ratio: 2.4 }, { thr_db: -20, ratio: 2.4 }],
      exciterMax: 0.05, satMax: 0.60, deEssMin: null, lufs: -14,
    },
    bright: {
      name: "bright", label: "Яркий / воздушный",
      deglass: 0.7, glassHz: [2400, 3800], glassCutDb: 2.2,
      warm: 0.15, warmEven: 0.15, airDb: 3.0, airHz: 9500,
      punch: 0.45, width: 1.22, widthLowKeep: 0.85,
      gains: { "5000-12000": 1.0 }, comp: null,
      exciterMax: 0.22, satMax: 0.40, deEssMin: null, lufs: -13.5,
    },
    club: {
      name: "club", label: "Клуб / плотный",
      deglass: 0.5, glassHz: [2300, 3600], glassCutDb: 1.8,
      warm: 0.30, warmEven: 0.25, airDb: 1.5, airHz: 10500,
      punch: 0.50, width: 1.10, widthLowKeep: 0.80,
      gains: { "20-60": 1.0, "60-150": 0.5 }, comp: [{ thr_db: -18, ratio: 2.6 }, { thr_db: -19, ratio: 2.2 }, { thr_db: -20, ratio: 2.4 }],
      exciterMax: 0.15, satMax: 0.55, deEssMin: null, lufs: -10.5,
    },
  };

  // ------------------------------------------------------------- helpers --
  function planar64From(x) {
    const y = E.planar64(x.n);
    y.L.set(x.L); y.R.set(x.R);
    return y;
  }
  function planar64To32(x) {
    return { L: new Float32Array(x.L), R: new Float32Array(x.R), n: x.n };
  }
  function cascade2(b, a) {
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    const b0 = b[0], b1 = b[1], b2 = b[2], a1 = a[1], a2 = a[2];
    return v => {
      const y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = v; y2 = y1; y1 = y;
      return y;
    };
  }
  function bandPass(sig, sr, lo, hi) {
    const [bl, al] = E.butter2(sr, clip(lo, 10, sr * 0.49), "hp");
    const [bh, ah] = E.butter2(sr, clip(hi, 10, sr * 0.49), "lp");
    const h1 = cascade2(bl, al), h2 = cascade2(bl, al);
    const l1 = cascade2(bh, ah), l2 = cascade2(bh, ah);
    const out = new Float64Array(sig.length);
    for (let i = 0; i < sig.length; i++) out[i] = l2(l1(h2(h1(sig[i]))));
    return out;
  }
  function mkEnv(sig, sr, attackMs, releaseMs) {
    const ca = attackMs <= 0 ? 1 : 1 - Math.exp(-1000 / (attackMs * sr));
    const cr = releaseMs <= 0 ? 1 : 1 - Math.exp(-1000 / (releaseMs * sr));
    const env = new Float64Array(sig.length);
    let g = Math.abs(sig[0] || 0);
    for (let i = 0; i < sig.length; i++) {
      const v = Math.abs(sig[i]);
      g += (v < g ? ca : cr) * (v - g);
      env[i] = g;
    }
    return env;
  }

  function biquadCut(x, sr, fc, cutDb, q) {
    const [b, a] = E.biquadPeaking(sr, fc, cutDb, q);
    const tL = cascade2(b, a), tR = cascade2(b, a);
    for (let i = 0; i < x.n; i++) { x.L[i] = tL(x.L[i]); x.R[i] = tR(x.R[i]); }
    return x;
  }

  function warmSat(x, sr, prof, k) {
    const w = clip(prof.warm * k, 0, 0.9);
    if (w < 0.02) return x;
    const even = clip(prof.warmEven * k, 0, 0.6);
    const drive = 1 + w * 2.0;
    const dcA = Math.exp(-2 * Math.PI * 10 / sr);
    let sL = 0, sR = 0, pL = 0, pR = 0;
    for (let i = 0; i < x.n; i++) {
      let l = x.L[i] * drive, r = x.R[i] * drive;
      l = Math.tanh(l) / Math.max(drive, EPS) * 1.05;
      r = Math.tanh(r) / Math.max(drive, EPS) * 1.05;
      l += even * 0.35 * Math.tanh(x.L[i] * drive * 2) * Math.abs(Math.tanh(x.L[i] * drive * 2));
      r += even * 0.35 * Math.tanh(x.R[i] * drive * 2) * Math.abs(Math.tanh(x.R[i] * drive * 2));
      const ol = l - pL + dcA * sL; pL = l; sL = ol;
      const or_ = r - pR + dcA * sR; pR = r; sR = or_;
      x.L[i] = ol; x.R[i] = or_;
    }
    return x;
  }

  function airShelf(x, sr, prof, k) {
    const g = prof.airDb * k;
    if (Math.abs(g) < 0.15) return x;
    const [b, a] = E.biquadShelf(sr, prof.airHz, g, 0.8, true);
    const tL = cascade2(b, a), tR = cascade2(b, a);
    for (let i = 0; i < x.n; i++) { x.L[i] = tL(x.L[i]); x.R[i] = tR(x.R[i]); }
    return x;
  }

  function punch(x, sr, prof, k, mul) {
    const amt = clip(prof.punch * k * (mul == null ? 1 : mul), 0, 1.2);
    if (amt < 0.03) return x;
    const f = mkEnv(x.L, sr, 0.8, 0.8), s = mkEnv(x.L, sr, 8, 220);
    const f2 = mkEnv(x.R, sr, 0.8, 0.8), s2 = mkEnv(x.R, sr, 8, 220);
    const MAXU = 0.10; // не более +10% линейного усиления атаки
    for (let i = 0; i < x.n; i++) {
      const rel = f[i] / (s[i] + EPS) + f2[i] / (s2[i] + EPS);
      const tr = clip((rel - 1.35) / 1.2, 0, 1);
      const g = 1 + amt * MAXU * tr;
      x.L[i] *= g; x.R[i] *= g;
    }
    return x;
  }

  // Мультибэнд-стерео через side: низ плотнее (lowKeep), верх шире (×1.5)
  function widthSimple(x, sr, prof, k) {
    const wEff = 1 + (prof.width - 1) * clip(k, 0, 1.4);
    const lowKeep = 1 + (prof.widthLowKeep - 1) * clip(k, 0, 1.4);
    const fLow = 180, fHi = 4500;
    const sL = new Float64Array(x.n), mm = new Float64Array(x.n);
    for (let i = 0; i < x.n; i++) { mm[i] = (x.L[i] + x.R[i]) / 2; sL[i] = x.L[i] - mm[i]; }
    const loB = bandPass(sL, sr, 18, fLow);
    const midB = bandPass(sL, sr, fLow, fHi);
    const hiB = bandPass(sL, sr, fHi, sr * 0.48);
    for (let i = 0; i < x.n; i++) {
      const s = loB[i] * lowKeep + midB[i] * wEff + hiB[i] * wEff * 1.5;
      x.L[i] = mm[i] + s; x.R[i] = mm[i] - s;
    }
    return x;
  }

  // -------------------------------------------------------------- стемы ---
  function classifyStem(name, feats) {
    if (name === "drums") return { role: "drums", sub: "drums" };
    if (name === "bass") return { role: "bass", sub: "bass" };
    if (name === "vocals") return { role: "vocals", sub: "vocals" };
    const b = feats.balance_db || {};
    const g = kk => (b[kk] != null ? b[kk] : -99);
    const hiMid = g("2000-5000"), air = g("5000-12000"), low = g("60-150"), lowMid = g("150-400");
    if (hiMid > lowMid + 1.5 && hiMid > air + 3) return { role: "other", sub: "guitar", note: "гитароподобная (акцент 2–5 кГц)" };
    if (low > hiMid + 3.5 && lowMid > hiMid) return { role: "other", sub: "toms", note: "низкочастотные удары (томы/перкуссия)" };
    if (air > hiMid - 1 && feats.stereo_width_db > -20) return { role: "other", sub: "keys", note: "клавиши/пэды (широко, много верха)" };
    return { role: "other", sub: "keys", note: "гармоническая дорожка" };
  }

  const STEM_SHARE_TARGET = { drums: -14, bass: -10, vocals: -8, other: -12 };
  const STEM_GAIN_CAP = { drums: 4.0, bass: 6.0, vocals: 6.0, other: 6.0 };
  const SUB2ROLE = { drums: "drums", toms: "drums", bass: "bass", vocals: "vocals", guitar: "other", keys: "other" };

  function autoStemGain(fIn, mixLufs, name) {
    const cls = classifyStem(name, fIn);
    const role = SUB2ROLE[cls.sub] || cls.role;
    const rel = fIn.lufs - mixLufs;
    const target = STEM_SHARE_TARGET[role] + 2.5;   // запас +2.5 дБ
    const delta = target - rel;
    const gainDb = clip(delta, -8, STEM_GAIN_CAP[role]);
    const quiet = (rel + gainDb) < (STEM_SHARE_TARGET[role] - 1.0);
    const fixes = [];
    if (gainDb > 0.8) fixes.push("подтянул громкость: +" + gainDb.toFixed(1) + " dB (дорожка была тише нормы)");
    if (gainDb < -0.8) fixes.push("приглушил: " + gainDb.toFixed(1) + " dB (торчала из микса)");
    if (quiet) fixes.push("дорожка остаётся тише нормы и после подтяжки — очень тихая в исходнике");
    return { cls, role, rel, target, gainDb, quiet, fixes, extraGain: db2lin(gainDb) };
  }

  // Динамический «демаскировщик» стеклянного бэнда
  function deGlassFull(xIn, sr, profile, intensity) {
    const prof = typeof profile === "string" ? (PROFILES[profile] || PROFILES.auto) : profile;
    const k = clip(intensity == null ? 1 : intensity, 0, 1.5);
    const x = planar64From(xIn);
    const lo = prof.glassHz[0], hi = prof.glassHz[1];
    const gL = bandPass(x.L, sr, lo, hi), gR = bandPass(x.R, sr, lo, hi);
    const all = new Float64Array(x.n);
    for (let i = 0; i < x.n; i++) all[i] = (x.L[i] + x.R[i]) * 0.5;
    const envG = mkEnv(gL, sr, 8, 120), envA = mkEnv(all, sr, 8, 200);
    const maxCutDb = -prof.glassCutDb * k;
    const fc = Math.sqrt(lo * hi);
    const staticCutDb = maxCutDb * 0.25;
    for (let i = 0; i < x.n; i++) {
      const share = envG[i] / (envA[i] * 1.6 + EPS);
      const over = clip((share - 0.85) / 1.15, 0, 1);
      const duckDb = maxCutDb * over;
      const gain = duckDb > -0.05 ? 1 : db2lin(duckDb);
      x.L[i] += gL[i] * (gain - 1);
      x.R[i] += gR[i] * (gain - 1);
    }
    if (Math.abs(staticCutDb) > 0.1) biquadCut(x, sr, fc, staticCutDb, 1.0);
    return { x32: planar64To32(x), glassCut: staticCutDb };
  }

  const glassExcess = (feats) => {
    const b = feats.balance_db;
    return (b["2000-5000"] != null && b["400-2000"] != null) ? b["2000-5000"] - b["400-2000"] : null;
  };

  // Полная v2-обработка готового аудио (после основной цепочки)
  function enhanceV2(xIn, sr, profile, intensity, opts) {
    const prof = typeof profile === "string" ? (PROFILES[profile] || PROFILES.auto) : profile;
    const k = clip(intensity == null ? 1 : intensity, 0, 1.5);
    const stemMul = opts && opts.stemMul != null ? opts.stemMul : 1;
    const punchMul = opts && opts.punchMul != null ? opts.punchMul : 1;
    const x = planar64From(xIn);
    const dg = deGlassFull(x, sr, prof, k);
    const y = dg.x32;
    if (prof.warm > 0) warmSat(y, sr, prof, k * stemMul);
    airShelf(y, sr, prof, k * stemMul);
    if (prof.punch > 0) punch(y, sr, prof, k, punchMul);
    widthSimple(y, sr, prof, k);
    return y;
  }

  // ---------------------------------------------------------- main chain --
  function applyProfileToParams(p, prof) {
    if (prof.gains) {
      for (const kk of Object.keys(prof.gains)) {
        if (p.band_gains_db[kk] != null) p.band_gains_db[kk] = clip(p.band_gains_db[kk] + prof.gains[kk], -6, 6);
      }
    }
    if (prof.comp) {
      p.comp = p.comp.map((c, i) => {
        const o = prof.comp[i] || {};
        return {
          thr_db: o.thr_db != null ? o.thr_db : c.thr_db,
          ratio: o.ratio != null ? o.ratio : c.ratio,
          attack_ms: o.attack_ms != null ? o.attack_ms : c.attack_ms,
          release_ms: o.release_ms != null ? o.release_ms : c.release_ms,
        };
      });
    }
    if (prof.exciterMax != null && p.exciter) p.exciter.mix = Math.min(p.exciter.mix, prof.exciterMax);
    if (prof.satMax != null) p.saturation = Math.min(p.saturation, prof.satMax);
    if (prof.deEssMin != null && p.de_ess) p.de_ess.amount = Math.max(p.de_ess.amount, prof.deEssMin);
    if (prof.lufs != null) p.target_lufs = prof.lufs;
    return p;
  }

  async function processAsyncV2(xIn, sr, opts, onStatus) {
    opts = opts || {}; onStatus = onStatus || (() => {});
    const prof = PROFILES[opts.profile || "auto"] || PROFILES.auto;
    const k = clip(opts.intensity == null ? 1 : opts.intensity, 0, 1.5);
    const x = planar64From(xIn);
    onStatus("Анализ трека (v2)…");
    const featsIn = E.analyze(x, sr);
    const glassBefore = glassExcess(featsIn);
    const widthBefore = featsIn.stereo_width_db;

    const hasUser = !!opts.params;
    let p = hasUser ? E.sanitize(JSON.parse(JSON.stringify(opts.params))) : E.predict(featsIn, sr);
    if (hasUser) {
      p._user = true;
      if (prof.lufs != null) p.target_lufs = prof.lufs;
    } else {
      applyProfileToParams(p, prof);
    }
    const profKee = prof;
    const preDb = clip(-18 - featsIn.lufs, -20, 20);
    const g0 = db2lin(preDb);
    for (let i = 0; i < x.n; i++) { x.L[i] *= g0; x.R[i] *= g0; }

    const iters = opts.iters == null ? 6 : opts.iters;
    let best = null;
    for (let it = 0; it < iters; it++) {
      onStatus("Рендер и измерение: проход " + (it + 1) + "/" + iters);
      const y = E.safeRender(x, sr, p);
      const ft = E.analyze(y, sr);
      const L = E.loss(ft, p);
      if (!best || L < best.L) best = { L, y, ft, p: JSON.parse(JSON.stringify(p)) };
      if (it < iters - 1) {
        p = E.refineStep(JSON.parse(JSON.stringify(p)), ft);
        if (!hasUser) applyProfileToParams(p, profKee);
      }
      await new Promise(r => setTimeout(r, 0));
    }

    onStatus("Профиль звука: " + prof.label + " (сила " + Math.round(k * 100) + "%)…");
    let y2 = planar64From(best.y);
    const dg = deGlassFull(y2, sr, prof, k);
    y2 = dg.x32;
    if (prof.warm > 0) warmSat(y2, sr, prof, k);
    airShelf(y2, sr, prof, k);
    if (prof.punch > 0) punch(y2, sr, prof, k, 1);
    widthSimple(y2, sr, prof, k);

    const ft2 = E.analyze(y2, sr);
    const tgtLufs = best.p.target_lufs != null ? best.p.target_lufs : (prof.lufs != null ? prof.lufs : -14);
    let dr = clip(tgtLufs - ft2.lufs, -6, 6);
    if (Math.abs(dr) > 0.35) {
      const g = db2lin(dr);
      for (let i = 0; i < y2.n; i++) { y2.L[i] *= g; y2.R[i] *= g; }
    }
    E.limitTo(y2, sr, best.p.ceiling_dbtp != null ? best.p.ceiling_dbtp : -1);
    const ftF = E.analyze(y2, sr);

    const used = Object.assign({}, best.p);
    delete used._user;
    const glassAfter = glassExcess(ftF);
    const notes = [];
    if (glassBefore != null && glassAfter != null) {
      if (glassAfter < glassBefore - 0.3) notes.push("«стекло» 2–5 кГц снижено на " + (glassBefore - glassAfter).toFixed(1) + " dB относительно середины");
      else notes.push("«стекло» удержано (без роста относительно середины)");
    }
    if (prof.width > 1) notes.push("стерео-база: " + widthBefore.toFixed(1) + " → " + ftF.stereo_width_db.toFixed(1) + " dB");
    if (prof.warm > 0.3) notes.push("тёплая сатурация (2-я гармоника) включена");

    const result = {
      profile: prof.name, intensity: k,
      params: used,
      loss: best.L,
      before: E.publicFeats(featsIn),
      after: E.publicFeats(ftF),
      pre_gain_db: preDb,
      v2: {
        glass_before: glassBefore, glass_after: glassAfter,
        width_before: widthBefore, width_after: ftF.stereo_width_db,
        notes,
      },
    };
    return { result, y: y2 };
  }

  // --------------------------------------------------------------- stems --
  async function masterStemsAsyncV2(xIn, sr, onStatus, onStem, opts) {
    opts = opts || {}; onStatus = onStatus || (() => {});
    const prof = PROFILES[opts.profile || "auto"] || PROFILES.auto;
    const k = clip(opts.intensity == null ? 1 : opts.intensity, 0, 1.5) * 0.85;
    const stemIters = opts.stemIters == null ? 2 : opts.stemIters;
    onStatus("Разделение на дорожки…");
    const stems = await E.separateAsync(xIn, sr, onStatus);
    const mixLufs = E.lufsGated(xIn, sr);
    const out = {};
    const mix = E.planar32(xIn.n);
    const PUNCH_MUL = { drums: 1.25, bass: 0.7, vocals: 0.9, other: 1.0 };
    for (const name of ["drums", "bass", "vocals", "other"]) {
      onStatus("Мастеринг дорожки: " + name);
      const s = stems[name]; stems[name] = null;
      if (!s || !s.n) continue;
      const fIn = E.analyze(s, sr);
      const ag = autoStemGain(fIn, mixLufs, name);
      const preL = db2lin(clip(-20 - fIn.lufs + ag.gainDb, -20, 20));
      const sw32 = { L: new Float32Array(s.n), R: new Float32Array(s.n), n: s.n };
      for (let i = 0; i < s.n; i++) { sw32.L[i] = s.L[i] * preL; sw32.R[i] = s.R[i] * preL; }
      let pr = E.buildStemParams((ag.sub === "guitar" || ag.sub === "keys") ? "other" : name, fIn, 1.05);
      pr = E.sanitize(pr);
      let y = E.safeRender(sw32, sr, pr);
      let ft = E.analyze(y, sr);
      for (let i = 0; i < stemIters; i++) {
        pr = E.refineStep(JSON.parse(JSON.stringify(pr)), ft);
        y = E.safeRender(sw32, sr, pr);
        ft = E.analyze(y, sr);
        await new Promise(r => setTimeout(r, 0));
      }
      const y2raw = enhanceV2(y, sr, prof, k, { stemMul: 0.8, punchMul: PUNCH_MUL[name] || 1 });
      const y2 = E.planar64(y2raw.n);
      y2.L.set(y2raw.L); y2.R.set(y2raw.R);
      E.limitTo(y2, sr, -1);
      const yF = { L: new Float32Array(y2.L), R: new Float32Array(y2.R), n: y2.n };
      const ftF = E.analyze(yF, sr);
      const diag = {
        role: ag.role, sub: ag.cls.sub, note: ag.cls.note || "",
        rel_lufs: +ag.rel.toFixed(1), target_lufs: STEM_SHARE_TARGET[ag.role],
        gain_db: +ag.gainDb.toFixed(1), quiet: ag.quiet, fixes: ag.fixes,
      };
      out[name] = { raw: s, master: yF, params: pr, diag, before: E.publicFeats(fIn), after: E.publicFeats(ftF) };
      if (onStem) {
        await onStem(name, { raw: s, master: yF, params: pr, diag, before: fIn, after: ftF });
        out[name].raw = null; out[name].master = null;
      } else {
        for (let i = 0; i < s.n; i++) { mix.L[i] += yF.L[i]; mix.R[i] += yF.R[i]; }
      }
    }
    return { stems: out, mixLufs, mix };
  }

  async function processFullV2(xIn, sr, opts, onStatus) {
    const main = await processAsyncV2(xIn, sr, opts, onStatus);
    const res = main.result;
    if (opts && opts.stems) {
      const sm = await masterStemsAsyncV2(xIn, sr, onStatus, null, opts);
      res.stems = {};
      for (const nm of Object.keys(sm.stems)) {
        const st = sm.stems[nm];
        res.stems[nm] = { before: st.before, after: st.after, params: st.params, diag: st.diag };
      }
      main.sm = sm;
      res.stem_mix_lufs = sm.mixLufs;
    }
    main.result = res;
    return main;
  }

  return {
    PROFILES, enhanceV2, deGlassFull, classifyStem, autoStemGain,
    processAsyncV2, masterStemsAsyncV2, processFullV2,
    glassExcess, bandPass, mkEnv, widthSimple, punch, warmSat, airShelf, biquadCut,
    STEM_SHARE_TARGET, STEM_GAIN_CAP,
  };
});
