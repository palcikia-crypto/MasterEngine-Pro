/* MasterEngine — pure-JS DSP engine (browser + Node, no Python, no WASM downloads).
 * Faithful port of masterengine/*.py. Stereo planar float64, stem buffers float32.
 * Not bit-exact vs scipy by design: Butterworth approximated with RBJ BLT,
 * bandpass = HP+LP cascade, true peak = 4x Catmull-Rom (tight enough for -1 dBTP).
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.MasterEngine = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const EPS = 1e-12;
  const BANDS = [[20, 60], [60, 150], [150, 400], [400, 2000], [2000, 5000], [5000, 12000], [12000, 20000]];
  const BAND_KEYS = BANDS.map(b => b[0] + "-" + b[1]);
  const TARGET_LUFS = -14.0;
  const TARGET_CEILING_DBTP = -1.0;
  const TARGET_CREST_RANGE = [9.5, 12.5];
  const REFERENCE_BALANCE = {
    "20-60": -8.0, "60-150": -5.0, "150-400": -8.0, "400-2000": -7.5,
    "2000-5000": -9.5, "5000-12000": -13.0, "12000-20000": -20.0,
  };

  const tick = () => new Promise(r => setTimeout(r, 0));
  const db = x => 20 * Math.log10(Math.max(x, EPS));
  const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  function planar64(n) { return { L: new Float64Array(n), R: new Float64Array(n), n }; }
  function planar32(n) { return { L: new Float32Array(n), R: new Float32Array(n), n }; }
  function copyPlanar(x) {
    const y = planar64(x.n);
    y.L.set(x.L); y.R.set(x.R);
    return y;
  }
  function reGain(x, g) {
    const y = planar64(x.n);
    for (let i = 0; i < x.n; i++) { y.L[i] = x.L[i] * g; y.R[i] = x.R[i] * g; }
    return y;
  }
  function f32ToF64(x) {
    const y = planar64(x.n);
    y.L.set(x.L); y.R.set(x.R);
    return y;
  }

  // ------------------------------------------------------------------ FFT --
  const FT = { size: 0 };
  function ensureFFT(size) {
    if (FT.size === size) return FT;
    const perm = new Uint32Array(size);
    for (let i = 1, j = 0; i < size; i++) {
      let bit = size >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      perm[i] = j; perm[j] = i;
    }
    const cosT = new Float64Array(size / 2), sinT = new Float64Array(size / 2);
    for (let i = 0; i < size / 2; i++) {
      cosT[i] = Math.cos(2 * Math.PI * i / size);
      sinT[i] = Math.sin(2 * Math.PI * i / size);
    }
    FT.size = size; FT.perm = perm; FT.cosT = cosT; FT.sinT = sinT;
    return FT;
  }
  function fftInOut(re, im, n, inverse) {
    const t = ensureFFT(n), perm = t.perm, cosT = t.cosT, sinT = t.sinT;
    for (let i = 0; i < n; i++) {
      const j = perm[i];
      if (j > i) { let v = re[i]; re[i] = re[j]; re[j] = v; v = im[i]; im[i] = im[j]; im[j] = v; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1, step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0; k < half; k++) {
          const tw = k * step;
          const wr = cosT[tw], wi = inverse ? sinT[tw] : -sinT[tw];
          const a = i + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
    if (inverse) { const inv = 1 / n; for (let i = 0; i < n; i++) { re[i] *= inv; im[i] *= inv; } }
  }
  function nextPow2(n) { let p = 1; while (p < n) p <<= 1; return p; }

  // --------------------------------------------------------------- Filters --
  function biquadPeaking(sr, f, gainDb, q) {
    const A = Math.pow(10, gainDb / 40), w0 = 2 * Math.PI * f / sr;
    const alpha = Math.sin(w0) / (2 * q), c = Math.cos(w0);
    const b = [1 + alpha * A, -2 * c, 1 - alpha * A], a = [1 + alpha / A, -2 * c, 1 - alpha / A];
    return [b.map(v => v / a[0]), a.map(v => v / a[0])];
  }
  function biquadShelf(sr, f, gainDb, q, high) {
    const A = Math.pow(10, gainDb / 40), w0 = 2 * Math.PI * f / sr;
    const alpha = Math.sin(w0) / (2 * q), c = Math.cos(w0);
    const sq = 2 * Math.sqrt(A) * alpha;
    let b, a;
    if (high) {
      b = [A * ((A + 1) + (A - 1) * c + sq), -2 * A * ((A - 1) + (A + 1) * c), A * ((A + 1) + (A - 1) * c - sq)];
      a = [(A + 1) - (A - 1) * c + sq, 2 * ((A - 1) - (A + 1) * c), (A + 1) - (A - 1) * c - sq];
    } else {
      b = [A * ((A + 1) - (A - 1) * c + sq), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sq)];
      a = [(A + 1) + (A - 1) * c + sq, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sq];
    }
    return [b.map(v => v / a[0]), a.map(v => v / a[0])];
  }
  function butter2(sr, f, kind) {
    const w0 = 2 * Math.PI * f / sr, c = Math.cos(w0), alpha = Math.sin(w0) / Math.SQRT2;
    let b;
    if (kind === "lp") b = [(1 - c) / 2, 1 - c, (1 - c) / 2];
    else b = [(1 + c) / 2, -(1 + c), (1 + c) / 2];
    const a0 = 1 + alpha;
    return [b.map(v => v / a0), [0, -2 * c / a0, (1 - alpha) / a0]];
  }
  function applyBiquadCh(x, b, a) {
    const n = x.length, y = new Float64Array(n);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    const b0 = b[0], b1 = b[1], b2 = b[2], a1 = a[1], a2 = a[2];
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = xi; y2 = y1; y1 = yi;
      y[i] = yi;
    }
    return y;
  }
  function applyBiquad(x, b, a) {
    return { L: applyBiquadCh(x.L, b, a), R: applyBiquadCh(x.R, b, a), n: x.n };
  }
  function lr4Split(x, sr, freq, kind) {
    const fc = clip(freq, 10, sr * 0.49);
    const [b, a] = butter2(sr, fc, kind);
    return applyBiquad(applyBiquad(x, b, a), b, a);
  }

  // ------------------------------------------------- Dynamics (block-rate) --
  function blockEnvPeakArr(env, x, block) {
    const n = x.n, cnt = env.length, m = Math.min(cnt * block, n);
    for (let c = 0; c < cnt; c++) {
      const s = c * block;
      let peak = 0;
      const e = Math.min(s + block, m);
      for (let i = s; i < e; i++) {
        const al = Math.abs(x.L[i]); if (al > peak) peak = al;
        const ar = Math.abs(x.R[i]); if (ar > peak) peak = ar;
      }
      env[c] = peak;
    }
    return env;
  }
  function smoothCoef(ms, ctrlSr) { return ms <= 0 ? 1.0 : 1 - Math.exp(-1000 / (ms * ctrlSr)); }
  function smoothTargets(targets, attackMs, releaseMs, ctrlSr) {
    const out = new Float64Array(targets.length);
    let g = targets[0];
    const ca = smoothCoef(attackMs, ctrlSr), cr = smoothCoef(releaseMs, ctrlSr);
    for (let i = 0; i < targets.length; i++) {
      const c = targets[i] < g ? ca : cr;
      g += c * (targets[i] - g);
      out[i] = g;
    }
    return out;
  }
  function makeGainAt(gainLin, n, block) {
    const len = gainLin.length;
    if (!len) return () => 1;
    if (len === 1) return () => gainLin[0];
    return i => {
      const pos = (i - 0.5 * block) / block;
      if (pos <= 0) return gainLin[0];
      if (pos >= len - 1) return gainLin[len - 1];
      const j = pos | 0;
      return gainLin[j] + (pos - j) * (gainLin[j + 1] - gainLin[j]);
    };
  }
  function compGainCurve(env, ctrlSr, thrDb, ratio, attackMs, releaseMs, kneeDb) {
    const lin = 1 - 1 / ratio;
    const g = new Float64Array(env.length);
    for (let i = 0; i < env.length; i++) {
      const o = db(env[i]) - thrDb;
      if (o <= 0) g[i] = 0;
      else if (o <= kneeDb) g[i] = lin * o * o / (2 * kneeDb);
      else g[i] = lin * (o - kneeDb / 2);
    }
    const sm = smoothTargets(g, attackMs, releaseMs, ctrlSr);
    const gainLin = new Float64Array(env.length);
    for (let i = 0; i < env.length; i++) gainLin[i] = Math.pow(10, -sm[i] / 20);
    return gainLin;
  }

  function compress(x, sr, thrDb, ratio, attackMs, releaseMs, makeupDb, kneeDb, block) {
    block = block || 256; kneeDb = kneeDb == null ? 6 : kneeDb;
    const cnt = Math.max(1, Math.floor(x.n / block));
    const env = blockEnvPeakArr(new Float64Array(cnt), x, block);
    const gainLin = compGainCurve(env, sr / block, thrDb, ratio, attackMs, releaseMs, kneeDb);
    const mk = Math.pow(10, (makeupDb || 0) / 20);
    const gAt = makeGainAt(gainLin, x.n, block);
    for (let i = 0; i < x.n; i++) {
      const gi = gAt(i) * mk;
      x.L[i] *= gi; x.R[i] *= gi;
    }
    return x;
  }

  function limit(x, sr, ceilingDb, lookaheadMs, attackMs, releaseMs, block) {
    ceilingDb = ceilingDb == null ? -1 : ceilingDb;
    lookaheadMs = lookaheadMs == null ? 1.5 : lookaheadMs;
    attackMs = attackMs == null ? 0.05 : attackMs;
    releaseMs = releaseMs == null ? 80 : releaseMs;
    block = block || 32;
    const cnt = Math.max(1, Math.floor(x.n / block));
    const env = blockEnvPeakArr(new Float64Array(cnt), x, block);
    const ctrlSr = sr / block;
    const thr = Math.pow(10, ceilingDb / 20);
    const needDb = new Float64Array(cnt);
    for (let i = 0; i < cnt; i++) needDb[i] = Math.min(0, db(Math.min(1, thr / Math.max(env[i], EPS))));
    const nLook = Math.max(1, Math.round(lookaheadMs * ctrlSr));
    const shifted = new Float64Array(cnt);
    for (let i = 0; i < cnt; i++) shifted[i] = i >= nLook ? needDb[i - nLook] : 0;
    const sm = smoothTargets(shifted, attackMs, releaseMs, ctrlSr);
    const gainLin = new Float64Array(cnt);
    for (let i = 0; i < cnt; i++) {
      const s = Math.max(-sm[i], 0);
      gainLin[i] = Math.pow(10, Math.min(-s, needDb[i]) / 20);
    }
    const gAt = makeGainAt(gainLin, x.n, block);
    for (let i = 0; i < x.n; i++) {
      const gi = gAt(i);
      x.L[i] *= gi; x.R[i] *= gi;
    }
    return x;
  }

  // 4x Catmull-Rom interpolated peak (approximates sinc-over true peak within ~0.1 dB)
  function truePeakLin(x) {
    const n = x.n;
    let peak = 0;
    const scan = s => {
      for (let i = 1; i < n - 2; i++) {
        const a = s[i - 1], b = s[i], c = s[i + 1], d = s[i + 2];
        for (let f = 0; f < 4; f++) {
          const t = f / 4, t2 = t * t, t3 = t2 * t;
          const v = 0.5 * (2 * b + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
          const av = Math.abs(v); if (av > peak) peak = av;
        }
      }
    };
    scan(x.L); scan(x.R);
    for (let i = 0; i < n; i++) {
      const a = Math.abs(x.L[i]); if (a > peak) peak = a;
      const b = Math.abs(x.R[i]); if (b > peak) peak = b;
    }
    return db(peak);
  }
  function reGainIP(x, g) {
    for (let i = 0; i < x.n; i++) { x.L[i] *= g; x.R[i] *= g; }
    return x;
  }
  function limitTo(x, sr, ceilingDb) {
    limit(x, sr, ceilingDb);
    for (let k = 0; k < 4; k++) {
      const tp = truePeakLin(x);
      if (tp <= ceilingDb + 0.03) break;
      reGainIP(x, Math.pow(10, (ceilingDb - 0.05 - tp) / 20));
    }
    return x;
  }

  // ------------------------------------------------------------ Multiband --
  function casc2(b, a) { return [biquadStep(b, a), biquadStep(b, a)]; }
  function biquadStep(b, a) {
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    const b0 = b[0], b1 = b[1], b2 = b[2], a1 = a[1], a2 = a[2];
    return v => {
      const y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = v; y2 = y1; y1 = y;
      return y;
    };
  }
  function stageSet(k) { const s = []; for (let i = 0; i < k; i++) s.push(new Float64Array(4)); return s; }
  const ME_BLOCK = 65536;
  function runStages(src, scratch, n, coeffs, states) {
    let S = src;
    for (let s = 0; s < states.length; s++) {
      const o = states[s], D = scratch[s], ba = coeffs[s], b = ba[0], a = ba[1];
      const b0 = b[0], b1 = b[1], b2 = b[2], a1 = a[1], a2 = a[2];
      let x1 = o[0], x2 = o[1], y1 = o[2], y2 = o[3];
      for (let i = 0; i < n; i++) {
        const xi = S[i];
        const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = xi; y2 = y1; y1 = yi;
        D[i] = yi;
      }
      o[0] = x1; o[1] = x2; o[2] = y1; o[3] = y2;
      S = D;
    }
    return S;
  }
  function runStagesPP(src, sA, sB, n, coeffs, states) {
    let S = src;
    for (let s = 0; s < states.length; s++) {
      const o = states[s], D = (s & 1) === 0 ? sA : sB, ba = coeffs[s], b = ba[0], a = ba[1];
      const b0 = b[0], b1 = b[1], b2 = b[2], a1 = a[1], a2 = a[2];
      let x1 = o[0], x2 = o[1], y1 = o[2], y2 = o[3];
      for (let i = 0; i < n; i++) {
        const xi = S[i];
        const yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = xi; y2 = y1; y1 = yi;
        D[i] = yi;
      }
      o[0] = x1; o[1] = x2; o[2] = y1; o[3] = y2;
      S = D;
    }
    return S;
  }
  function gainInterp(A, last, pos) {
    if (pos <= 0) return A[0];
    if (pos >= last) return A[last];
    const j = pos | 0;
    return A[j] + (pos - j) * (A[j + 1] - A[j]);
  }
  function peakBlocks(env, arr, block) {
    const cnt = env.length, m = Math.min(cnt * block, arr.length);
    for (let c = 0; c < cnt; c++) {
      const s = c * block, e = Math.min(s + block, m);
      let p = 0;
      for (let i = s; i < e; i++) { const a = Math.abs(arr[i]); if (a > p) p = a; }
      if (p > env[c]) env[c] = p;
    }
  }
  function multibandCompress(x, sr, crossovers, bands) {
    crossovers = crossovers || [180, 2500];
    bands = bands || [
      { thr_db: -18, ratio: 2.5, attack_ms: 18, release_ms: 140 },
      { thr_db: -20, ratio: 2.0, attack_ms: 8, release_ms: 120 },
      { thr_db: -22, ratio: 2.5, attack_ms: 5, release_ms: 100 },
    ];
    const n = x.n;
    const x0 = clip(crossovers[0], 10, sr * 0.49), x1 = clip(crossovers[1], 10, sr * 0.49);
    const [blo, alo] = butter2(sr, x0, "lp");
    const [bhi, ahi] = butter2(sr, x0, "hp");
    const [mlo, amlo] = butter2(sr, x1, "lp");
    const [mhi, amhi] = butter2(sr, x1, "hp");
    const block = 256, cnt = Math.max(1, Math.floor(n / block)), ctrlSr = sr / block;
    const envLo = new Float64Array(cnt), envMid = new Float64Array(cnt), envHi = new Float64Array(cnt);
    const midFull = planar32(n);
    const t1 = new Float32Array(n), t2 = new Float32Array(n), t3 = new Float32Array(n);
    const bio = [[blo, alo], [blo, alo]], bhp = [[bhi, ahi], [bhi, ahi]];
    const mio = [[mlo, amlo], [mlo, amlo]], mhp = [[mhi, amhi], [mhi, amhi]];
    runStages(x.L, [t1, t2], n, bio, stageSet(2));
    runStages(x.R, [t1, t3], n, bio, stageSet(2));
    peakBlocks(envLo, t2, block); peakBlocks(envLo, t3, block);
    runStages(x.L, [t1, midFull.L], n, bhp, stageSet(2));
    runStages(x.R, [t1, midFull.R], n, bhp, stageSet(2));
    runStages(midFull.L, [t1, t2], n, mio, stageSet(2));
    peakBlocks(envMid, t2, block);
    runStages(midFull.R, [t1, t2], n, mio, stageSet(2));
    peakBlocks(envMid, t2, block);
    runStages(midFull.L, [t1, t2], n, mhp, stageSet(2));
    peakBlocks(envHi, t2, block);
    runStages(midFull.R, [t1, t2], n, mhp, stageSet(2));
    peakBlocks(envHi, t2, block);
    const gLoA = compGainCurve(envLo, ctrlSr, bands[0].thr_db, bands[0].ratio, bands[0].attack_ms, bands[0].release_ms, 6);
    const gMidA = compGainCurve(envMid, ctrlSr, bands[1].thr_db, bands[1].ratio, bands[1].attack_ms, bands[1].release_ms, 6);
    const gHiA = compGainCurve(envHi, ctrlSr, bands[2].thr_db, bands[2].ratio, bands[2].attack_ms, bands[2].release_ms, 6);
    const gL1 = gLoA.length - 1, gL2 = gMidA.length - 1, gL3 = gHiA.length - 1;
    const invB = 1 / block;
    const out = planar32(n);
    for (const chn of ["L", "R"]) {
      const src = x[chn], mid = midFull[chn], dst = out[chn];
      runStages(src, [t1, t3], n, bio, stageSet(2));
      runStages(mid, [t2, t1], n, mio, stageSet(2));
      runStages(mid, [t2, t2], n, mhp, stageSet(2));
      for (let i = 0; i < n; i++) {
        const pos = i * invB - 0.5;
        dst[i] = t3[i] * gainInterp(gLoA, gL1, pos)
               + t1[i] * gainInterp(gMidA, gL2, pos)
               + t2[i] * gainInterp(gHiA, gL3, pos);
      }
    }
    return out;
  }

  // ------------------------------------------- Saturation / exciter / ess --
  function saturate(x, drive, mix) {
    drive = drive == null ? 1.5 : drive; mix = mix == null ? 1 : mix;
    const gd = 1 + drive * 3, norm = Math.max(drive, EPS);
    for (let i = 0; i < x.n; i++) {
      const l = x.L[i], r = x.R[i];
      x.L[i] = l * (1 - mix) + Math.tanh(l * gd) / norm * mix;
      x.R[i] = r * (1 - mix) + Math.tanh(r * gd) / norm * mix;
    }
    return x;
  }
  function excite(x, sr, freq, drive, mix) {
    drive = drive == null ? 2.5 : drive;
    const [b, a] = butter2(sr, clip(freq == null ? 3000 : freq, 10, sr * 0.49), "hp");
    const dmin = Math.min(drive, 1.0);
    const sL = casc2(b, a), sR = casc2(b, a);
    for (let i = 0; i < x.n; i++) {
      const hL = sL[1](sL[0](x.L[i])), hR = sR[1](sR[0](x.R[i]));
      const harL = Math.tanh(Math.tanh(hL * drive) - hL * dmin) * 0.8;
      const harR = Math.tanh(Math.tanh(hR * drive) - hR * dmin) * 0.8;
      x.L[i] += harL * mix * 0.2;
      x.R[i] += harR * mix * 0.2;
    }
    return x;
  }
  function deEss(x, sr, freq, amount) {
    const [b, a] = butter2(sr, clip(freq == null ? 6500 : freq, 10, sr * 0.49), "hp");
    const block = 256, cnt = Math.max(1, Math.floor(x.n / block));
    const env = new Float64Array(cnt);
    const sL = casc2(b, a), sR = casc2(b, a);
    for (let i = 0; i < x.n; i++) {
      const hL = sL[1](sL[0](x.L[i])), hR = sR[1](sR[0](x.R[i]));
      const c = i < block * cnt ? i / block | 0 : cnt - 1;
      const pL = Math.abs(hL), pR = Math.abs(hR);
      if (pL > env[c]) env[c] = pL;
      if (pR > env[c]) env[c] = pR;
    }
    const gainLin = compGainCurve(env, sr / block, -26, 4, 1, 60, 6);
    const gAt = makeGainAt(gainLin, x.n, block);
    const sL2 = casc2(b, a), sR2 = casc2(b, a);
    for (let i = 0; i < x.n; i++) {
      const hL = sL2[1](sL2[0](x.L[i])), hR = sR2[1](sR2[0](x.R[i]));
      const g = gAt(i);
      x.L[i] += hL * (g - 1) * amount;
      x.R[i] += hR * (g - 1) * amount;
    }
    return x;
  }

  // ---------------------------------------------------------------- Stereo --
  function convolveSameOLA(sig, kernel) {
    const n = sig.length, k = kernel.length, start = (k - 1) >> 1;
    const chunk = 16384;
    const F = nextPow2(chunk + k);
    const kre = new Float64Array(F), kim = new Float64Array(F);
    kre.set(kernel);
    fftInOut(kre, kim, F, false);
    const re = new Float64Array(F), im = new Float64Array(F);
    const out = new Float64Array(n);
    for (let c0 = 0; c0 < n; c0 += chunk) {
      const cLen = Math.min(chunk, n - c0);
      re.fill(0); im.fill(0);
      for (let i = 0; i < cLen; i++) re[i] = sig[c0 + i];
      fftInOut(re, im, F, false);
      for (let i = 0; i < F; i++) {
        const rr = re[i] * kre[i] - im[i] * kim[i];
        im[i] = re[i] * kim[i] + im[i] * kre[i];
        re[i] = rr;
      }
      fftInOut(re, im, F, true);
      for (let i = 0; i < cLen + k - 1; i++) {
        const j = c0 + i - start;
        if (j >= 0 && j < n) out[j] += re[i];
      }
    }
    return out;
  }
  function widen(x, sr, width, bassMonoHz) {
    const n = x.n;
    for (let i = 0; i < n; i++) {
      const l = x.L[i], r = x.R[i];
      x.L[i] = l - r;
      x.R[i] = (l + r) / 2;
    }
    const side = x.L, mid = x.R;
    if (width !== 1) {
      const amt = width - 1, fl = Math.floor(0.012 * sr);
      let wide;
      if (fl > 3) {
        const wl = Math.min(fl >> 1, 8192);
        const k = new Float64Array(wl);
        let sum = 0;
        for (let i = 0; i < wl; i++) { const v = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (2 * fl)); k[i] = v; sum += v; }
        for (let i = 0; i < wl; i++) k[i] = k[i] / sum * amt;
        wide = convolveSameOLA(side, k);
      } else {
        wide = new Float64Array(n);
        for (let i = 0; i < n; i++) wide[i] = side[i] * amt;
      }
      for (let i = 0; i < n; i++) {
        const t = 0.5 * (side[i] + wide[i]);
        const l = mid[i] - t, r = mid[i] + t;
        side[i] = l; mid[i] = r;
      }
    } else {
      for (let i = 0; i < n; i++) {
        const t = 0.5 * side[i];
        const l = mid[i] - t, r = mid[i] + t;
        side[i] = l; mid[i] = r;
      }
    }
    return x;
  }

  // -------------------------------------------------------------- Metering --
  const _kwCache = {};
  function kCoeffs(sr) {
    if (!_kwCache[sr]) {
      const [bs, as_] = biquadShelf(sr, 1681.97, 3.999843853973363, 0.7071752369554196, true);
      const [bh, ah] = butter2(sr, 38.0, "hp");
      _kwCache[sr] = [[bs, as_], [bh, ah], [bh, ah]];
    }
    return _kwCache[sr];
  }
  function kWeightScan(x, sr, collectHops) {
    const coeffs = kCoeffs(sr);
    const stL = stageSet(3), stR = stageSet(3);
    const sc = [new Float64Array(ME_BLOCK), new Float64Array(ME_BLOCK), new Float64Array(ME_BLOCK)];
    const hop = Math.max(1, Math.floor(0.100 * sr));
    let sum = 0, hopAcc = 0, hopCnt = 0, hopIdx = 0;
    const ring = new Float64Array(4);
    const hopE = collectHops ? [] : null;
    for (let off = 0; off < x.n; off += ME_BLOCK) {
      const len = Math.min(ME_BLOCK, x.n - off);
      const oL = runStages(x.L.subarray(off, off + len), sc, len, coeffs, stL);
      for (let i = 0; i < len; i++) {
        const v = oL[i];
        sum += v * v;
        if (collectHops) {
          hopAcc += v * v;
          if (++hopCnt === hop) { hopCnt = 0; ring[hopIdx & 3] = hopAcc; hopAcc = 0; if (hopIdx >= 3) hopE.push((ring[0] + ring[1] + ring[2] + ring[3]) / (4 * hop)); hopIdx++; }
        }
      }
      const oR = runStages(x.R.subarray(off, off + len), sc, len, coeffs, stR);
      for (let i = 0; i < len; i++) {
        const v = oR[i];
        sum += v * v;
        if (collectHops) {
          hopAcc += v * v;
          if (++hopCnt === hop) { hopCnt = 0; ring[hopIdx & 3] = hopAcc; hopAcc = 0; if (hopIdx >= 3) hopE.push((ring[0] + ring[1] + ring[2] + ring[3]) / (4 * hop)); hopIdx++; }
        }
      }
    }
    return { sum, hopE };
  }
  function lufs(x, sr) {
    const r = kWeightScan(x, sr, false);
    return -0.691 + 10 * Math.log10(r.sum / x.n + EPS);
  }
  function lufsGated(x, sr) {
    const size = Math.floor(0.400 * sr);
    if (x.n < size) return lufs(x, sr);
    const { hopE } = kWeightScan(x, sr, true);
    const lg = new Array(hopE.length);
    for (let i = 0; i < hopE.length; i++) lg[i] = -0.691 + 10 * Math.log10(hopE[i] + EPS);
    const absIdx = [];
    for (let i = 0; i < hopE.length; i++) if (lg[i] > -70) absIdx.push(i);
    if (!absIdx.length) return -70;
    let meanAbs = 0;
    for (let i = 0; i < absIdx.length; i++) meanAbs += hopE[absIdx[i]];
    meanAbs /= absIdx.length;
    const rel = -0.691 + 10 * Math.log10(meanAbs + EPS);
    const gIdx = [];
    for (let i = 0; i < absIdx.length; i++) if (lg[absIdx[i]] > rel - 10) gIdx.push(absIdx[i]);
    if (!gIdx.length) {
      let acc = 0;
      for (let i = 0; i < absIdx.length; i++) acc += lg[absIdx[i]];
      return acc / absIdx.length;
    }
    let acc = 0;
    for (let i = 0; i < gIdx.length; i++) acc += hopE[gIdx[i]];
    return -0.691 + 10 * Math.log10(acc / gIdx.length + EPS);
  }
  function bandEnergyDb(x, sr, fmin, fmax) {
    fmin = Math.max(fmin, 12); fmax = Math.min(fmax, sr / 2 - 1);
    const [b1, a1] = butter2(sr, fmin, "hp");
    const [b2, a2] = butter2(sr, fmax, "lp");
    const coeffs = [[b1, a1], [b1, a1], [b2, a2], [b2, a2]];
    const stL = stageSet(4), stR = stageSet(4);
    const sc = [new Float64Array(ME_BLOCK), new Float64Array(ME_BLOCK), new Float64Array(ME_BLOCK), new Float64Array(ME_BLOCK)];
    let s = 0;
    for (let off = 0; off < x.n; off += ME_BLOCK) {
      const len = Math.min(ME_BLOCK, x.n - off);
      const oL = runStages(x.L.subarray(off, off + len), sc, len, coeffs, stL);
      for (let i = 0; i < len; i++) s += oL[i] * oL[i];
      const oR = runStages(x.R.subarray(off, off + len), sc, len, coeffs, stR);
      for (let i = 0; i < len; i++) s += oR[i] * oR[i];
    }
    return db(Math.sqrt(s / (2 * x.n) + EPS));
  }
  function crestFactor(x) {
    let peak = 0, s = 0;
    for (let i = 0; i < x.n; i++) {
      const a = Math.abs(x.L[i]); if (a > peak) peak = a;
      const b = Math.abs(x.R[i]); if (b > peak) peak = b;
      s += x.L[i] * x.L[i] + x.R[i] * x.R[i];
    }
    const rms = Math.sqrt(s / (2 * x.n));
    return db(peak + EPS) - db(rms + EPS);
  }
  function stereoWidth(x) {
    let em = 0, es = 0;
    for (let i = 0; i < x.n; i++) {
      const m = (x.L[i] + x.R[i]) / 2, s = x.L[i] - x.R[i];
      em += m * m; es += s * s;
    }
    em = Math.sqrt(em / x.n + EPS); es = Math.sqrt(es / x.n + EPS);
    return db(es / em);
  }
  function spectralCentroid(x, sr) {
    const N = 4096, hop = N / 2;
    if (x.n < N) return 1000;
    const w = new Float64Array(N);
    for (let i = 0; i < N; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1));
    let accW = 0, accE = 0, frames = 0;
    const re = new Float64Array(N), im = new Float64Array(N);
    const seg = new Float64Array(N);
    for (let off = 0; off + N <= x.n; off += hop) {
      let mean = 0;
      for (let i = 0; i < N; i++) { const m = (x.L[off + i] + x.R[off + i]) / 2; seg[i] = m; mean += m; }
      mean /= N;
      for (let i = 0; i < N; i++) { re[i] = (seg[i] - mean) * w[i]; im[i] = 0; }
      fftInOut(re, im, N, false);
      for (let k = 1; k < N / 2; k++) {
        const f = k * sr / N;
        if (f <= 40 || f >= 16000) continue;
        const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
        accW += f * mag; accE += mag;
      }
      frames++;
    }
    if (!frames || accE <= 0) return 1000;
    return accW / (accE + 1e-12);
  }
  function maxAbsCh(a) { let m = 0; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m; }
  function analyze(x, sr) {
    const peak = Math.max(maxAbsCh(x.L), maxAbsCh(x.R));
    const balance = {};
    for (const bnd of BANDS) balance[bnd[0] + "-" + bnd[1]] = bandEnergyDb(x, sr, bnd[0], bnd[1]);
    return {
      lufs: lufsGated(x, sr),
      peak_dbtp: truePeakLin(x),
      crest_db: crestFactor(x),
      stereo_width_db: stereoWidth(x),
      balance_db: balance,
      peak_lin: peak,
      centroid_hz: spectralCentroid(x, sr),
    };
  }
  function balanceVector(feats) {
    const v = BAND_KEYS.map(k => feats.balance_db[k]);
    let mx = -Infinity;
    for (const vi of v) if (vi > mx) mx = vi;
    const out = new Array(v.length);
    for (let i = 0; i < v.length; i++) out[i] = v[i] - mx;
    return out;
  }
  function publicFeats(f) { const g = Object.assign({}, f); delete g.peak_lin; return g; }

  // ------------------------------------------------------------------- ML --
  function defaultParams() {
    return {
      hpf_hz: 30.0,
      band_gains_db: {},
      xover_hz: [180.0, 2500.0],
      comp: [
        { thr_db: -20.0, ratio: 2.0, attack_ms: 20.0, release_ms: 160.0 },
        { thr_db: -20.0, ratio: 2.0, attack_ms: 10.0, release_ms: 120.0 },
        { thr_db: -22.0, ratio: 2.0, attack_ms: 6.0, release_ms: 100.0 },
      ],
      saturation: 0.2,
      exciter: { freq: 3200.0, mix: 0.12 },
      de_ess: { freq: 6800.0, amount: 0.2 },
      stereo_width: 1.15,
      bass_mono_hz: 140.0,
      target_lufs: TARGET_LUFS,
      ceiling_dbtp: TARGET_CEILING_DBTP,
    };
  }
  const experience = [];
  function nearest(feats, k) {
    if (!experience.length) return null;
    return experience.map(e => {
      let d = 0;
      for (const key of BAND_KEYS) {
        const ev = (e.features.balance_db || {})[key];
        if (ev == null) { d += 1e6; continue; }
        d += Math.pow(ev - feats.balance_db[key], 2);
      }
      d = Math.sqrt(d) + Math.abs(feats.lufs - e.features.lufs) * 0.25;
      return { d, e };
    }).sort((a, b) => a.d - b.d).slice(0, k || 5).map(s => s.e);
  }
  function sanitizeBandGains(p) {
    if (!p.band_gains_db || typeof p.band_gains_db !== "object") p.band_gains_db = {};
    const bg = {};
    for (const k of BAND_KEYS) if (k in p.band_gains_db) bg[k] = clip(+p.band_gains_db[k] || 0, -6, 6);
    p.band_gains_db = bg;
    return p;
  }
  function sanitize(p) {
    sanitizeBandGains(p);
    p.stereo_width = clip(p.stereo_width == null ? 1.15 : +p.stereo_width, 0.6, 1.7);
    p.saturation = clip(p.saturation == null ? 0.2 : +p.saturation, 0, 0.8);
    p.exciter = p.exciter || { freq: 3200, mix: 0.1 };
    p.exciter.mix = clip(+p.exciter.mix || 0, 0, 0.35);
    p.de_ess = p.de_ess || { freq: 6800, amount: 0.2 };
    p.de_ess.amount = clip(+p.de_ess.amount || 0, 0, 0.8);
    p.hpf_hz = clip(p.hpf_hz == null ? 30 : +p.hpf_hz, 16, 80);
    p.target_lufs = clip(p.target_lufs == null ? TARGET_LUFS : +p.target_lufs, -24, -6);
    p.ceiling_dbtp = clip(p.ceiling_dbtp == null ? TARGET_CEILING_DBTP : +p.ceiling_dbtp, -3, -0.1);
    if (!Array.isArray(p.comp) || p.comp.length !== 3) p.comp = defaultParams().comp;
    if (!Array.isArray(p.xover_hz) || p.xover_hz.length !== 2) p.xover_hz = [180, 2500];
    return p;
  }
  function predict(feats, sr) {
    const params = defaultParams();
    for (const k of BAND_KEYS) params.band_gains_db[k] = 0.0;
    const crest = feats.crest_db;
    if (crest > 13.5) for (const c of params.comp) { c.ratio = Math.min(c.ratio + 1, 4); c.thr_db -= 2; }
    else if (crest < 8) for (const c of params.comp) { c.ratio = Math.max(c.ratio - 0.8, 1.1); c.thr_db += 3; }
    const cur = balanceVector(feats);
    for (let i = 0; i < BAND_KEYS.length; i++) {
      const key = BAND_KEYS[i], ref = REFERENCE_BALANCE[key];
      params.band_gains_db[key] = clip(-(cur[i] - ref) * 0.45, -6, 6);
    }
    const w = feats.stereo_width_db;
    if (w < -22) params.stereo_width = 1.3;
    else if (w > -12) params.stereo_width = 1.0;
    const centroid = feats.centroid_hz == null ? 1000 : feats.centroid_hz;
    if (centroid < 900) {
      params.exciter.mix = 0.18;
      params.band_gains_db["2000-5000"] += 0.5;
      params.band_gains_db["5000-12000"] += 0.5;
    } else if (centroid > 3200) {
      params.exciter.mix = 0.06;
      params.band_gains_db["2000-5000"] -= 0.5;
      params.band_gains_db["150-400"] += 0.5;
    }
    const near = nearest(feats);
    if (near) {
      for (const key of BAND_KEYS) {
        const vals = near.filter(nt => nt.params.band_gains_db && key in nt.params.band_gains_db).map(nt => nt.params.band_gains_db[key]);
        if (vals.length) {
          const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
          params.band_gains_db[key] = 0.7 * params.band_gains_db[key] + 0.3 * avg;
        }
      }
      for (let j = 0; j < 3; j++) {
        const thr = near.filter(nt => nt.params.comp && nt.params.comp.length === 3).map(nt => nt.params.comp[j].thr_db);
        const rat = near.filter(nt => nt.params.comp && nt.params.comp.length === 3).map(nt => nt.params.comp[j].ratio);
        if (thr.length) {
          params.comp[j].thr_db = 0.7 * params.comp[j].thr_db + 0.3 * (thr.reduce((a, b) => a + b, 0) / thr.length);
          params.comp[j].ratio = 0.7 * params.comp[j].ratio + 0.3 * (rat.reduce((a, b) => a + b, 0) / rat.length);
        }
      }
    }
    return sanitize(params);
  }
  function refineStep(params, featsOut, lr) {
    lr = lr == null ? 0.7 : lr;
    const p = JSON.parse(JSON.stringify(params));
    const dLufs = p.target_lufs - featsOut.lufs;
    p._output_gain_db = clip((p._output_gain_db || 0) + dLufs * lr, -18, 12);
    if (!p._user) {
      const cur = balanceVector(featsOut);
      for (let i = 0; i < BAND_KEYS.length; i++) {
        const key = BAND_KEYS[i];
        if (!(key in p.band_gains_db)) continue;
        const err = cur[i] - (REFERENCE_BALANCE[key] != null ? REFERENCE_BALANCE[key] : -10);
        p.band_gains_db[key] = clip(p.band_gains_db[key] - err * 0.35 * lr, -6, 6);
      }
    }
    const crest = featsOut.crest_db, lo = TARGET_CREST_RANGE[0], hi = TARGET_CREST_RANGE[1];
    const d = 0.15 * lr;
    if (crest < lo) for (const c of p.comp) { c.ratio = clip(c.ratio + d, 1.1, 4); c.thr_db = clip(c.thr_db - d * 4, -30, -10); }
    else if (crest > hi + 0.5) for (const c of p.comp) { c.ratio = clip(c.ratio + d, 1.1, 4); c.thr_db = clip(c.thr_db - d * 6, -30, -10); c.release_ms = clip(c.release_ms * 0.92, 60, 250); }
    return p;
  }

  // -------------------------------------------------------------- Chain ----
  function copyPreGain(x, g) {
    const y = planar32(x.n);
    for (let i = 0; i < x.n; i++) { y.L[i] = x.L[i] * g; y.R[i] = x.R[i] * g; }
    return y;
  }
  function chainCoeffs(sr, p, x) {
    const coeffs = [butter2(sr, p.hpf_hz, "hp")];
    for (const k of BAND_KEYS) {
      if (!(k in p.band_gains_db)) continue;
      const fmin = +k.split("-")[0], fmax = +k.split("-")[1];
      const g = p.band_gains_db[k];
      if (Math.abs(g) < 0.05) continue;
      if (fmin <= 60) coeffs.push(biquadShelf(sr, Math.sqrt(fmin * fmax) * 1.2, g, 0.9, false));
      else if (fmax >= 12000) coeffs.push(biquadShelf(sr, 10000, g, 0.8, true));
      else coeffs.push(biquadPeaking(sr, Math.sqrt(fmin * fmax), g, fmax < 300 ? 1.1 : 0.9));
    }
    return coeffs;
  }
  function applyChain(x, sr, p) {
    const n = x.n;
    const coeffs = chainCoeffs(sr, p, x);
    const K = coeffs.length;
    const sA = new Float32Array(n), sB = new Float32Array(n);
    const buf = planar32(n);
    buf.L.set(runStagesPP(x.L, sA, sB, n, coeffs, stageSet(K)));
    buf.R.set(runStagesPP(x.R, sA, sB, n, coeffs, stageSet(K)));
    let y = multibandCompress(buf, sr, p.xover_hz, p.comp);
    if (p.saturation > 0.01) saturate(y, p.saturation * 3, 0.5);
    if (p.exciter.mix > 0.005) excite(y, sr, p.exciter.freq == null ? 3000 : p.exciter.freq, 2.5, p.exciter.mix);
    if (p.de_ess.amount > 0.005) deEss(y, sr, p.de_ess.freq == null ? 6500 : p.de_ess.freq, p.de_ess.amount);
    widen(y, sr, p.stereo_width, p.bass_mono_hz);
    reGainIP(y, Math.pow(10, (p._output_gain_db || 0) / 20));
    limitTo(y, sr, p.ceiling_dbtp);
    return y;
  }

  const STREAM_MIN_SEC = 120, CS = 65536;
  let _streamScratch = null;
  function streamScratch() {
    if (_streamScratch) return _streamScratch;
    const z = () => new Float32Array(CS);
    _streamScratch = { t1: z(), t2: z(), t3: z(), t4: z(), loL: z(), loR: z(), mfL: z(), mfR: z(),
      m2L: z(), m2R: z(), h2L: z(), h2R: z(), eL: z(), eR: z(), dL: z(), dR: z(),
      seg: new Float32Array(CS + 8320) };
    return _streamScratch;
  }
  function gainPush(car, env, b) {
    const o = db(env[b]) - car.thr;
    let t;
    if (o <= 0) t = 0;
    else if (o <= car.knee) t = car.lin * o * o / (2 * car.knee);
    else t = car.lin * (o - car.knee / 2);
    let g;
    if (car.filled === 0) g = t;
    else g = car.g + (t < car.g ? car.ca : car.cr) * (t - car.g);
    car.g = g; car.filled = b + 1;
    return Math.pow(10, -g / 20);
  }
  function streamRender(x, sr, p) {
    const n = x.n;
    const out = planar32(n);
    const S = streamScratch();
    const x0 = clip(p.xover_hz[0] == null ? 180 : +p.xover_hz[0], 10, sr * 0.49);
    const x1 = clip(p.xover_hz[1] == null ? 2500 : +p.xover_hz[1], 10, sr * 0.49);
    const fil = butter2(sr, x0, "lp"), fih = butter2(sr, x0, "hp");
    const fml = butter2(sr, x1, "lp"), fmh = butter2(sr, x1, "hp");
    const bio = [[fil[0], fil[1]], [fil[0], fil[1]]], bhp = [[fih[0], fih[1]], [fih[0], fih[1]]];
    const mio = [[fml[0], fml[1]], [fml[0], fml[1]]], mhp = [[fmh[0], fmh[1]], [fmh[0], fmh[1]]];
    const eq = chainCoeffs(sr, p, x);
    const K = eq.length;
    const stEqL = stageSet(K), stEqR = stageSet(K);
    const stLoL = stageSet(2), stLoR = stageSet(2), stMfL = stageSet(2), stMfR = stageSet(2);
    const stM2L = stageSet(2), stM2R = stageSet(2), stH2L = stageSet(2), stH2R = stageSet(2);
    const stExL = stageSet(2), stExR = stageSet(2), stDeL = stageSet(2), stDeR = stageSet(2);
    const exFreq = p.exciter && p.exciter.freq != null ? +p.exciter.freq : 3000;
    const exa = butter2(sr, clip(exFreq, 10, sr * 0.49), "hp");
    const exo = [[exa[0], exa[1]], [exa[0], exa[1]]];
    const deFreq = p.de_ess && p.de_ess.freq != null ? +p.de_ess.freq : 6500;
    const dea = butter2(sr, clip(deFreq, 10, sr * 0.49), "hp");
    const dco = [[dea[0], dea[1]], [dea[0], dea[1]]];
    const exDr = 2.5, exDmin = Math.min(exDr, 1.0), exMix = p.exciter ? +p.exciter.mix || 0 : 0;
    const deAmt = p.de_ess ? +p.de_ess.amount || 0 : 0;

    const block = 256, invB = 1 / block, ctrlSr = sr / block;
    const cnt = Math.max(1, Math.floor(n / block));
    const envLo = new Float64Array(cnt), envMid = new Float64Array(cnt), envHi = new Float64Array(cnt), envDe = new Float64Array(cnt);
    const cgLo = new Float64Array(cnt), cgMid = new Float64Array(cnt), cgHi = new Float64Array(cnt), cgDe = new Float64Array(cnt);
    const mkCar = (thrDb, ratio, attackMs, releaseMs) => ({ thr: thrDb, knee: 6, lin: 1 - 1 / ratio, ca: smoothCoef(attackMs, ctrlSr), cr: smoothCoef(releaseMs, ctrlSr), g: 0, filled: 0 });
    const carLo = mkCar(p.comp[0].thr_db, p.comp[0].ratio, p.comp[0].attack_ms, p.comp[0].release_ms);
    const carMid = mkCar(p.comp[1].thr_db, p.comp[1].ratio, p.comp[1].attack_ms, p.comp[1].release_ms);
    const carHi = mkCar(p.comp[2].thr_db, p.comp[2].ratio, p.comp[2].attack_ms, p.comp[2].release_ms);
    const carDe = mkCar(-26, 4, 1, 60);

    const width = p.stereo_width, amt = width - 1, flW = Math.floor(0.012 * sr);
    const useConv = width !== 1 && flW > 3;
    const wl = useConv ? Math.min(flW >> 1, 8192) : 0;
    const kern = useConv ? new Float64Array(wl) : null;
    if (useConv) {
      let sum = 0;
      for (let i = 0; i < wl; i++) { const v = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (2 * flW)); kern[i] = v; sum += v; }
      for (let i = 0; i < wl; i++) kern[i] = kern[i] / sum * amt;
    }
    const startW = useConv ? (wl - 1) >> 1 : 0;
    const cap = CS + wl + 64;
    const qS = new Float32Array(cap), qM = new Float32Array(cap);
    const qPL = new Float32Array(cap), qPR = new Float32Array(cap);

    const invL = 1 / 32, ctrlL = sr / 32;
    const nLook = Math.max(1, Math.round(1.5 * ctrlL));
    const cntL = Math.max(1, Math.floor(n / 32));
    const envLm = new Float64Array(cntL), gLim = new Float64Array(cntL);
    const ca_l = smoothCoef(0.05, ctrlL), cr_l = smoothCoef(80, ctrlL);
    const thrL = Math.pow(10, (p.ceiling_dbtp == null ? -1 : +p.ceiling_dbtp) / 20);
    let gLc = 0, limBi = 0, accL = 0, accR = 0, limDone = 0;
    let jOut = 0, wrNext = 0;
    const gOut = Math.pow(10, (p._output_gain_db || 0) / 20);

    for (let c0 = 0; c0 < n; c0 += CS) {
      const c1 = Math.min(n, c0 + CS), len = c1 - c0;
      const A = S.t3, AR = S.t4;
      const cL = [], cR = [];
      for (let i = 0; i < K; i++) { cL.push(i % 3 === 2 ? A : (i % 3 === 0 ? S.t1 : S.t2)); cR.push(i % 3 === 2 ? AR : (i % 3 === 0 ? S.t1 : S.t2)); }
      const rL = runStages(x.L.subarray(c0, c1), cL, len, eq, stEqL);
      if (rL !== A) A.set(rL.subarray(0, len));
      const rR = runStages(x.R.subarray(c0, c1), cR, len, eq, stEqR);
      if (rR !== AR) AR.set(rR.subarray(0, len));

      const loLv = S.loL, loRv = S.loR, mfLv = S.mfL, mfRv = S.mfR;
      runStages(A, [S.t1, loLv], len, bio, stLoL);
      runStages(A, [S.t2, mfLv], len, bhp, stMfL);
      runStages(AR, [S.t1, loRv], len, bio, stLoR);
      runStages(AR, [S.t2, mfRv], len, bhp, stMfR);
      runStages(mfLv, [S.t1, S.m2L], len, mio, stM2L);
      runStages(mfLv, [S.t2, S.h2L], len, mhp, stH2L);
      runStages(mfRv, [S.t1, S.m2R], len, mio, stM2R);
      runStages(mfRv, [S.t2, S.h2R], len, mhp, stH2R);
      let bi = (c0 / block) | 0, bE = Math.min((bi + 1) * block, n);
      let pLo = 0, pMid = 0, pHi = 0;
      const m2Lv = S.m2L, m2Rv = S.m2R, h2Lv = S.h2L, h2Rv = S.h2R;
      for (let i = 0; i < len; i++) {
        let u = Math.abs(loLv[i]), v = Math.abs(loRv[i]); if (v > u) u = v; if (u > pLo) pLo = u;
        u = Math.abs(m2Lv[i]); v = Math.abs(m2Rv[i]); if (v > u) u = v; if (u > pMid) pMid = u;
        u = Math.abs(h2Lv[i]); v = Math.abs(h2Rv[i]); if (v > u) u = v; if (u > pHi) pHi = u;
        if (c0 + i + 1 === bE) {
          const bc = bi < cnt ? bi : cnt - 1;
          if (pLo > envLo[bc]) envLo[bc] = pLo;
          if (pMid > envMid[bc]) envMid[bc] = pMid;
          if (pHi > envHi[bc]) envHi[bc] = pHi;
          bi++; bE = Math.min((bi + 1) * block, n); pLo = 0; pMid = 0; pHi = 0;
        }
      }
      const bStart = (c0 / block) | 0, bEnd = Math.min(bi, cnt);
      for (let b = bStart; b < bEnd; b++) {
        cgLo[b] = gainPush(carLo, envLo, b);
        cgMid[b] = gainPush(carMid, envMid, b);
        cgHi[b] = gainPush(carHi, envHi, b);
      }
      const dLast = bEnd - 1;
      const oL = out.L.subarray(c0, c1), oRR = out.R.subarray(c0, c1);
      for (let i = 0; i < len; i++) {
        const pos = (c0 + i) * invB - 0.5;
        oL[i] = loLv[i] * gainInterp(cgLo, dLast, pos) + m2Lv[i] * gainInterp(cgMid, dLast, pos) + h2Lv[i] * gainInterp(cgHi, dLast, pos);
        oRR[i] = loRv[i] * gainInterp(cgLo, dLast, pos) + m2Rv[i] * gainInterp(cgMid, dLast, pos) + h2Rv[i] * gainInterp(cgHi, dLast, pos);
      }
      if (p.saturation > 0.01) {
        const drv = p.saturation * 3, gd = 1 + drv * 3, norm = Math.max(drv, EPS);
        for (let i = 0; i < len; i++) {
          const l = oL[i], r = oRR[i];
          oL[i] = l * 0.5 + Math.tanh(l * gd) / norm * 0.5;
          oRR[i] = r * 0.5 + Math.tanh(r * gd) / norm * 0.5;
        }
      }
      if (exMix > 0.005) {
        const eLv = S.eL, eRv = S.eR;
        runStages(oL, [S.t1, eLv], len, exo, stExL);
        runStages(oRR, [S.t2, eRv], len, exo, stExR);
        for (let i = 0; i < len; i++) {
          const vl = eLv[i], vr = eRv[i];
          oL[i] += (Math.tanh(Math.tanh(vl * exDr) - vl * exDmin) * 0.8) * exMix * 0.2;
          oRR[i] += (Math.tanh(Math.tanh(vr * exDr) - vr * exDmin) * 0.8) * exMix * 0.2;
        }
      }
      if (deAmt > 0.005) {
        const dLv = S.dL, dRv = S.dR;
        runStages(oL, [S.t1, dLv], len, dco, stDeL);
        runStages(oRR, [S.t2, dRv], len, dco, stDeR);
        let dbi = (c0 / block) | 0, dbE = Math.min((dbi + 1) * block, n), pDe = 0;
        for (let i = 0; i < len; i++) {
          const u = Math.abs(dLv[i]), v = Math.abs(dRv[i]); const pk = u > v ? u : v; if (pk > pDe) pDe = pk;
          if (c0 + i + 1 === dbE) {
            const bc = dbi < cnt ? dbi : cnt - 1;
            if (pDe > envDe[bc]) envDe[bc] = pDe;
            dbi++; dbE = Math.min((dbi + 1) * block, n); pDe = 0;
          }
        }
        const dStart = (c0 / block) | 0, dEnd = Math.min(dbi, cnt);
        for (let b = dStart; b < dEnd; b++) cgDe[b] = gainPush(carDe, envDe, b);
        const ddLast = dEnd - 1;
        if (dEnd > dStart) {
          for (let i = 0; i < len; i++) {
            const pos = (c0 + i) * invB - 0.5;
            const gg = gainInterp(cgDe, ddLast, pos) - 1;
            oL[i] += dLv[i] * gg * deAmt;
            oRR[i] += dRv[i] * gg * deAmt;
          }
        }
      }
      for (let i = 0; i < len; i++) {
        const sSlot = (c0 + i) % cap;
        qS[sSlot] = oL[i] - oRR[i];
        qM[sSlot] = (oL[i] + oRR[i]) / 2;
      }
      let cv = null;
      if (useConv) {
        for (let p = 0; p < wl; p++) { const m = c0 - wl + p; S.seg[p] = m < 0 ? 0 : qS[m % cap]; }
        for (let i = 0; i < len; i++) S.seg[wl + i] = qS[(c0 + i) % cap];
        cv = convolveSameOLA(S.seg.subarray(0, wl + len), kern);
      }
      const arrEnd = c1;
      while (jOut <= arrEnd - 1 - startW && jOut < n) {
        const sSlot = jOut % cap;
        const sd = qS[sSlot], md = qM[sSlot];
        let tp;
        if (width === 1) tp = 0.5 * sd;
        else if (useConv) tp = 0.5 * (sd + cv[jOut - c0 + wl]);
        else tp = 0.5 * (sd + sd * amt);
        qPL[sSlot] = (md - tp) * gOut;
        qPR[sSlot] = (md + tp) * gOut;
        const al = Math.abs(qPL[sSlot]), ar = Math.abs(qPR[sSlot]);
        if (al > accL) accL = al;
        if (ar > accR) accR = ar;
        if (jOut + 1 === (limBi + 1) * 32) {
          if (limBi < cntL) {
            envLm[limBi] = Math.max(accL, accR);
            const nd = Math.min(0, db(Math.min(1, thrL / Math.max(envLm[limBi], EPS))));
            const sh = limBi >= nLook ? Math.min(0, db(Math.min(1, thrL / Math.max(envLm[limBi - nLook], EPS)))) : 0;
            gLc += (sh < gLc ? ca_l : cr_l) * (sh - gLc);
            gLim[limBi] = Math.pow(10, Math.min(gLc, nd) / 20);
            limDone = limBi + 1; limBi++; accL = 0; accR = 0;
          }
        }
        jOut++;
      }
      if (limDone > 1) {
        const stop = Math.min(n, limDone * 32 - 16);
        while (wrNext < stop) {
          const g = gainInterp(gLim, limDone - 1, wrNext * invL - 0.5);
          const sSlot = wrNext % cap;
          out.L[wrNext] = qPL[sSlot] * g;
          out.R[wrNext] = qPR[sSlot] * g;
          wrNext++;
        }
      }
    }
    const c0Last = Math.floor((n - 1) / CS) * CS, c1Last = n;
    let cv2 = null;
    if (useConv && jOut < n) {
      for (let p = 0; p < wl; p++) { const m = c0Last - wl + p; S.seg[p] = m < 0 ? 0 : qS[m % cap]; }
      for (let i = 0; i < c1Last - c0Last; i++) S.seg[wl + i] = qS[(c0Last + i) % cap];
      cv2 = convolveSameOLA(S.seg.subarray(0, wl + c1Last - c0Last), kern);
    }
    while (jOut < n) {
      const sSlot = jOut % cap;
      const sd = qS[sSlot], md = qM[sSlot];
      let tp;
      if (width === 1) tp = 0.5 * sd;
      else if (useConv) tp = 0.5 * (sd + cv2[jOut - c0Last + wl]);
      else tp = 0.5 * (sd + sd * amt);
      qPL[sSlot] = (md - tp) * gOut;
      qPR[sSlot] = (md + tp) * gOut;
      const al = Math.abs(qPL[sSlot]), ar = Math.abs(qPR[sSlot]);
      if (al > accL) accL = al;
      if (ar > accR) accR = ar;
      if (jOut + 1 === (limBi + 1) * 32) {
        if (limBi < cntL) {
          envLm[limBi] = Math.max(accL, accR);
          const nd = Math.min(0, db(Math.min(1, thrL / Math.max(envLm[limBi], EPS))));
          const sh = limBi >= nLook ? Math.min(0, db(Math.min(1, thrL / Math.max(envLm[limBi - nLook], EPS)))) : 0;
          gLc += (sh < gLc ? ca_l : cr_l) * (sh - gLc);
          gLim[limBi] = Math.pow(10, Math.min(gLc, nd) / 20);
          limDone = limBi + 1; limBi++; accL = 0; accR = 0;
        }
      }
      jOut++;
    }
    while (wrNext < n) {
      const g = gainInterp(gLim, cntL - 1, wrNext * invL - 0.5);
      const sSlot = wrNext % cap;
      out.L[wrNext] = qPL[sSlot] * g;
      out.R[wrNext] = qPR[sSlot] * g;
      wrNext++;
    }
    for (let k2 = 0; k2 < 4; k2++) {
      const tpv = truePeakLin(out);
      if (tpv <= (p.ceiling_dbtp == null ? -1 : p.ceiling_dbtp) + 0.03) break;
      reGainIP(out, Math.pow(10, ((p.ceiling_dbtp == null ? -1 : p.ceiling_dbtp) - 0.05 - tpv) / 20));
    }
    for (let i = 0; i < n; i++) {
      if (!isFinite(out.L[i])) out.L[i] = 0;
      if (!isFinite(out.R[i])) out.R[i] = 0;
    }
    return out;
  }

  const renderOnce = (x, sr, p) => (x.n >= STREAM_MIN_SEC * sr ? streamRender(x, sr, p) : applyChain(x, sr, p));
  function safeRender(x, sr, p) {
    let y = renderOnce(x, sr, p);
    let rx = 0, ry = 0;
    for (let i = 0; i < x.n; i++) { rx += x.L[i] * x.L[i] + x.R[i] * x.R[i]; ry += y.L[i] * y.L[i] + y.R[i] * y.R[i]; }
    const ref = Math.sqrt(rx / (2 * x.n)) + 1e-12, cur = Math.sqrt(ry / (2 * x.n));
    if (!isFinite(cur) || cur > ref * 12) {
      const p2 = JSON.parse(JSON.stringify(p));
      p2._output_gain_db = (p2._output_gain_db || 0) - 12;
      y = renderOnce(x, sr, p2);
    }
    for (let i = 0; i < y.n; i++) {
      if (!isFinite(y.L[i])) y.L[i] = 0;
      if (!isFinite(y.R[i])) y.R[i] = 0;
    }
    return y;
  }
  function loss(featsOut, p) {
    const cur = balanceVector(featsOut);
    let err = 0;
    for (let i = 0; i < BAND_KEYS.length; i++) {
      const k = BAND_KEYS[i];
      const ref = REFERENCE_BALANCE[k] != null ? REFERENCE_BALANCE[k] : -12;
      const w = ref < -17.5 ? 0.5 : 1.0;
      err += w * Math.pow(cur[i] - ref, 2);
    }
    const dLufs = Math.pow(featsOut.lufs - p.target_lufs, 2) * 1.5;
    const crest = featsOut.crest_db;
    const dCrest = Math.pow(Math.max(0, TARGET_CREST_RANGE[0] - crest) + Math.max(0, crest - TARGET_CREST_RANGE[1] - 1.0), 2);
    return err * 0.05 + dLufs + dCrest * 0.5;
  }

  async function processAsync(xIn, sr, opts, onStatus) {
    opts = opts || {};
    onStatus = onStatus || (() => {});
    let x = xIn;
    onStatus("Анализ трека…");
    const featsIn = analyze(x, sr);
    let p;
    if (opts.params) {
      p = sanitize(JSON.parse(JSON.stringify(opts.params)));
      p.target_lufs = opts.params.target_lufs == null ? TARGET_LUFS : opts.params.target_lufs;
      p._user = true;
    } else {
      p = predict(featsIn, sr);
    }
    const preDb = clip(-18 - featsIn.lufs, -20, 20);
    x = copyPreGain(x, Math.pow(10, preDb / 20));
    const iters = opts.iters == null ? 6 : opts.iters;
    let best = null;
    for (let it = 0; it < iters; it++) {
      onStatus("Рендер и измерение: проход " + (it + 1) + "/" + iters);
      const y = safeRender(x, sr, p);
      const ft = analyze(y, sr);
      const L = loss(ft, p);
      if (!best || L < best.L) best = { L, y, ft, p: JSON.parse(JSON.stringify(p)) };
      if (it < iters - 1) p = refineStep(JSON.parse(JSON.stringify(p)), ft);
      await tick();
    }
    const presetParams = predict(featsIn, sr);
    const used = Object.assign({}, best.p);
    delete used._user;
    experience.push({ features: Object.assign({}, publicFeats(featsIn)), params: JSON.parse(JSON.stringify(used)) });
    if (experience.length > 128) experience.shift();
    return {
      result: {
        preset_params: presetParams,
        params: used,
        loss: best.L,
        before: publicFeats(featsIn),
        after: publicFeats(best.ft),
        pre_gain_db: preDb,
      },
      y: best.y,
    };
  }
  // --------------------------------------------------------------- Stems ---
  const N_FFT = 4096, HOP = 1024, CHUNK = 262144, OVERLAP = 32768;
  const STEM_ORDER = ["drums", "bass", "vocals", "other"];

  function softBandMaskF(freqs, lo, hi) {
    const soft = 0.35, m = new Float64Array(freqs.length);
    for (let i = 0; i < freqs.length; i++) {
      const f = freqs[i];
      const a = clip((lo - f) / (lo * soft + 1), -60, 60);
      const b = clip((f - hi) / (hi * soft + 1), -60, 60);
      m[i] = (1 / (1 + Math.exp(a))) * (1 / (1 + Math.exp(b)));
    }
    return m;
  }
  function medianFilterAxis(data, F, T, axis, size) {
    const out = new Float64Array(F * T);
    const half = size >> 1, win = new Float64Array(size);
    if (axis === 1) {
      for (let f = 0; f < F; f++) {
        const base = f * T;
        for (let t = 0; t < T; t++) {
          let cnt = 0;
          for (let k = -half; k <= half; k++) {
            const tt = t + k;
            if (tt >= 0 && tt < T) win[cnt++] = data[base + tt];
          }
          out[base + t] = medianOf(win, cnt);
        }
      }
    } else {
      for (let f = 0; f < F; f++) {
        const base = f * T;
        for (let t = 0; t < T; t++) {
          let cnt = 0;
          for (let k = -half; k <= half; k++) {
            const ff = f + k;
            if (ff >= 0 && ff < F) win[cnt++] = data[ff * T + t];
          }
          out[base + t] = medianOf(win, cnt);
        }
      }
    }
    return out;
  }
  function medianOf(win, cnt) {
    const a = Array.prototype.slice.call(win, 0, cnt).sort((x, y) => x - y);
    return a[cnt >> 1];
  }

  async function separateAsync(x, sr, onStatus) {
    onStatus = onStatus || (() => {});
    const n = x.n, ramp = OVERLAP / 2;
    const win = new Float64Array(ramp);
    for (let i = 0; i < ramp; i++) win[i] = Math.sqrt(0.5 - 0.5 * Math.cos(Math.PI * i / ramp));
    const outs = {};
    for (const nm of STEM_ORDER) outs[nm] = planar32(n);
    const weights = new Float64Array(n);
    const nSteps = Math.max(1, Math.ceil(n / CHUNK));
    const F = N_FFT / 2 + 1;
    const w = new Float64Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N_FFT);
    const fRe = new Float64Array(F);
    for (let k = 0; k < F; k++) fRe[k] = k * sr / N_FFT;
    const midsF = softBandMaskF(fRe, 220, 8000);
    const lowsF = softBandMaskF(fRe, 20, 190);
    let pos = 0, step = 0;
    while (pos < n) {
      step++;
      onStatus("Разделение на дорожки: блок " + step + "/" + nSteps);
      const end = Math.min(pos + CHUNK, n);
      const lo = Math.max(0, pos - OVERLAP), hi = Math.min(n, end + OVERLAP);
      const segLen = hi - lo;
      const env = new Float64Array(segLen);
      env.fill(1);
      if (lo > 0) for (let i = 0; i < ramp; i++) env[i] *= win[i];
      if (hi < n) for (let i = 0; i < ramp; i++) env[segLen - ramp + i] *= win[i];
      const pad = N_FFT / 2;
      const paddedLen = segLen + 2 * pad;
      const T = Math.floor((paddedLen - N_FFT) / HOP) + 1;
      const segL = new Float64Array(paddedLen), segR = new Float64Array(paddedLen);
      for (let i = 0; i < segLen; i++) { segL[pad + i] = x.L[lo + i]; segR[pad + i] = x.R[lo + i]; }
      const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
      const re2 = new Float64Array(N_FFT), im2 = new Float64Array(N_FFT);
      const ZL = new Float64Array(F * T * 2), ZR = new Float64Array(F * T * 2);
      const mag = new Float64Array(F * T);
      for (let t = 0; t < T; t++) {
        const off = t * HOP;
        for (let i = 0; i < N_FFT; i++) {
          re[i] = segL[off + i] * w[i]; im[i] = 0;
          re2[i] = segR[off + i] * w[i]; im2[i] = 0;
        }
        fftInOut(re, im, N_FFT, false);
        fftInOut(re2, im2, N_FFT, false);
        for (let k = 0; k < F; k++) {
          const idx = k * T + t;
          ZL[idx * 2] = re[k]; ZL[idx * 2 + 1] = im[k];
          ZR[idx * 2] = re2[k]; ZR[idx * 2 + 1] = im2[k];
          mag[idx] = (Math.hypot(re[k], im[k]) + Math.hypot(re2[k], im2[k])) / 2;
        }
      }
      const harm = medianFilterAxis(mag, F, T, 1, 17);
      const perc = medianFilterAxis(mag, F, T, 0, 17);
      const masks = {};
      for (const nm of STEM_ORDER) masks[nm] = new Float64Array(F * T);
      for (let k = 0; k < F; k++) {
        const m = midsF[k], l = lowsF[k];
        const base = k * T;
        for (let t = 0; t < T; t++) {
          const idx = base + t;
          const h2 = harm[idx] * harm[idx], p2 = perc[idx] * perc[idx];
          const h = h2 / (h2 + p2 + 1e-12);
          const drums = 1 - h;
          const vocals = h * m;
          const bass = h * l * clip(1 - m, 0, 1);
          const other = clip(1 - drums - vocals - bass, 0, 1);
          const M = masks;
          M.drums[idx] = drums; M.vocals[idx] = vocals; M.bass[idx] = bass; M.other[idx] = other;
        }
      }
      for (const nm of STEM_ORDER) {
        const buf = outs[nm];
        for (let sIdx = 0; sIdx < 2; sIdx++) {
          const src = sIdx === 0 ? ZL : ZR;
          const dst = sIdx === 0 ? buf.L : buf.R;
          const y = new Float64Array(paddedLen);
          const w2 = new Float64Array(paddedLen);
          const reI = new Float64Array(N_FFT), imI = new Float64Array(N_FFT);
          for (let t = 0; t < T; t++) {
            for (let k = 0; k < F; k++) {
              const idx = k * T + t, b = idx * 2, mask = masks[nm][idx];
              const pr = src[b] * mask, pi = src[b + 1] * mask;
              if (k === 0) { reI[0] = pr; imI[0] = 0; }
              else if (k === F - 1) { reI[N_FFT / 2] = pr; imI[N_FFT / 2] = 0; }
              else {
                reI[k] = pr; imI[k] = pi;
                reI[N_FFT - k] = pr; imI[N_FFT - k] = -pi;
              }
            }
            fftInOut(reI, imI, N_FFT, true);
            const off = t * HOP;
            for (let i = 0; i < N_FFT; i++) { y[off + i] += reI[i] * w[i]; w2[off + i] += w[i] * w[i]; }
          }
          for (let i = 0; i < segLen; i++) {
            const denom = w2[pad + i] > 1e-9 ? w2[pad + i] : 1;
            dst[lo + i] += (y[pad + i] / denom) * env[i];
          }
        }
      }
      for (let i = lo; i < hi; i++) weights[i] += env[i - lo];
      pos = end;
      await tick();
    }
    for (const nm of STEM_ORDER) {
      const b = outs[nm];
      for (let i = 0; i < n; i++) {
        const wgt = Math.max(weights[i], 1e-6);
        b.L[i] /= wgt; b.R[i] /= wgt;
      }
    }
    return outs;
  }

  const STEM_REFERENCE = {
    drums: { comp_thr: -14, comp_ratio: 2.0, saturation: 0.25, exciter_mix: 0.2,
      balance: { "20-60": -7.5, "60-150": -3.0, "150-400": -5.0, "400-2000": -3.0, "2000-5000": -5.0, "5000-12000": -8.0, "12000-20000": -16.0 } },
    bass: { comp_thr: -16, comp_ratio: 2.5, saturation: 0.2, exciter_mix: 0.06,
      balance: { "20-60": -6.0, "60-150": -2.0, "150-400": -5.0 } },
    vocals: { comp_thr: -18, comp_ratio: 2.2, saturation: 0.12, exciter_mix: 0.16,
      balance: { "400-2000": -3.0, "2000-5000": -5.0, "5000-12000": -9.0 } },
    other: { comp_thr: -20, comp_ratio: 1.8, saturation: 0.1, exciter_mix: 0.1,
      balance: { "2000-5000": -5.0, "5000-12000": -8.0 } },
  };
  function buildStemParams(name, feats, width) {
    width = width == null ? 1.0 : width;
    const p = {
      hpf_hz: { drums: 28, bass: 20, vocals: 90, other: 120 }[name] || 40,
      band_gains_db: {},
      xover_hz: [220, 4500],
      comp: [],
      saturation: 0.15,
      exciter: { freq: 3000, mix: 0.1 },
      de_ess: { freq: 7000, amount: name === "vocals" ? 0.25 : 0 },
      stereo_width: width,
      bass_mono_hz: name === "bass" ? 0 : 120,
      target_lufs: -14,
      ceiling_dbtp: -1,
    };
    const ref = STEM_REFERENCE[name] || {};
    const cur = balanceVector(feats);
    const base = {};
    for (const k of BAND_KEYS) base[k] = 0;
    for (let i = 0; i < BAND_KEYS.length; i++) {
      const k = BAND_KEYS[i];
      if (ref.balance && k in ref.balance && k in base) base[k] = clip((cur[i] - ref.balance[k]) * -0.4, -6, 6);
    }
    p.band_gains_db = base;
    const cthr = ref.comp_thr != null ? ref.comp_thr : -18;
    const crat = ref.comp_ratio != null ? ref.comp_ratio : 2.0;
    p.comp = [];
    for (let j = 0; j < 3; j++) p.comp.push({ thr_db: cthr, ratio: crat, attack_ms: 15, release_ms: 140 });
    p.saturation = ref.saturation != null ? ref.saturation : 0.15;
    p.exciter.mix = ref.exciter_mix != null ? ref.exciter_mix : 0.1;
    return p;
  }

  async function masterStemsAsync(x, sr, onStatus, onStem) {
    onStatus = onStatus || (() => {});
    const stems = await separateAsync(x, sr, onStatus);
    const out = {}, before = {}, after = {};
    for (const name of STEM_ORDER) {
      onStatus("Мастеринг дорожки: " + name);
      const s = stems[name];
      stems[name] = null;
      const fIn = analyze(s, sr);
      const preDb = clip(-22 - fIn.lufs, -20, 20);
      const sw = copyPreGain(s, Math.pow(10, preDb / 20));
      let p = buildStemParams(name, fIn);
      let y = applyChain(sw, sr, p);
      let ft = analyze(y, sr);
      for (let i = 0; i < 2; i++) {
        p = refineStep(JSON.parse(JSON.stringify(p)), ft);
        y = safeRender(sw, sr, p);
        ft = analyze(y, sr);
        await tick();
      }
      out[name] = { raw: s, master: y, params: p };
      before[name] = fIn; after[name] = ft;
      if (onStem) { await onStem(name, { raw: s, master: y, params: p, before: fIn, after: ft }); out[name].raw = null; out[name].master = null; }
    }
    return { stems: out, before, after };
  }
  // ------------------------------------------------------------------ WAV --
  function encodeWav(x, sr) {
    const n = x.n, dataLen = n * 2 * 2;
    const buf = new ArrayBuffer(44 + dataLen);
    const dv = new DataView(buf);
    const W = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
    W(0, "RIFF"); dv.setUint32(4, 36 + dataLen, true); W(8, "WAVE");
    W(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
    dv.setUint16(22, 2, true); dv.setUint32(24, Math.round(sr), true);
    dv.setUint32(28, Math.round(sr) * 4, true); dv.setUint16(32, 4, true); dv.setUint16(34, 16, true);
    W(36, "data"); dv.setUint32(40, dataLen, true);
    let off = 44;
    for (let i = 0; i < n; i++) {
      const l = Math.max(-1, Math.min(1, x.L[i])), r = Math.max(-1, Math.min(1, x.R[i]));
      dv.setInt16(off, l < 0 ? l * 32768 : l * 32767, true); off += 2;
      dv.setInt16(off, r < 0 ? r * 32768 : r * 32767, true); off += 2;
    }
    return new Uint8Array(buf);
  }

  return {
    BANDS, BAND_KEYS, TARGET_LUFS, TARGET_CEILING_DBTP, TARGET_CREST_RANGE,
    REFERENCE_BALANCE, planar64, planar32, copyPlanar, reGain, f32ToF64,
    biquadPeaking, biquadShelf, butter2, applyBiquad, lr4Split,
    compress, limit, limitTo, truePeak: truePeakLin, multibandCompress,
    saturate, excite, deEss, widen,
    lufs, lufsGated, crestFactor, stereoWidth, bandEnergyDb, spectralCentroid,
    analyze, balanceVector, publicFeats,
    predict, sanitize, refineStep, applyChain, safeRender, loss,
    processAsync, separateAsync, masterStemsAsync, buildStemParams, STEM_REFERENCE, streamRender,
    fftInOut, ensureFFT, medianFilterAxis, softBandMaskF,
    encodeWav,
  };
});
