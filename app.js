/* MasterEngine web panel — UI driving the engine web worker (installable PWA). */
if ("serviceWorker" in navigator && (location.protocol === "https:" || ["localhost", "127.0.0.1"].includes(location.hostname))) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
const $ = s => document.querySelector(s);

let worker = null, ready = false;
let beat = 0;               // request sequence
let lastRes = null, blobs = {};
let audioData = null, sr = 0, nSamples = 0, fileName = "";

const status = (msg, err, done, pct) => {
  const el = $("#status");
  el.textContent = msg;
  el.className = err ? "err" : "";
  const sp = $("#spin");
  sp.className = err ? "spin off" : (done ? "spin done" : (pct == null ? "spin" : "spin"));
  sp.textContent = done ? "✓" : "";
  if (pct != null) $("#pbarfill").style.width = pct + "%";
  else if (done) $("#pbarfill").style.width = "100%";
};

function ensureWorker() {
  if (worker) return;
  worker = new Worker("engine-worker.js");
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.t === "status") status(m.msg, false, false, m.pct);
    else if (m.t === "ready") {
      ready = true;
      $("#stemsOn").disabled = false;
      status("Движок готов — бросай трек, обработаю сам", false, true, 100);
    } else if (m.t === "result") {
      for (const u of Object.values(urlCache)) { try { URL.revokeObjectURL(u); } catch (_) {} }
      urlCache = {};
      blobs = {};
      for (const [p, buf] of Object.entries(m.files)) blobs[p] = buf;
      lastRes = m.res;
      status("Готово! Жми A ⇄ B и слушай разницу в наушниках. loss = " + m.res.loss.toFixed(2), false, true, 100);
      show();
    } else if (m.t === "error") {
      status("Ошибка: " + m.msg, true);
    }
  };
  worker.onerror = (e) => status("Worker: " + e.message, true);
}
ensureWorker();

const drop = $("#drop");
drop.onclick = () => ready && $("#file").click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add("on"); };
drop.ondragleave = () => drop.classList.remove("on");
drop.ondrop = e => { e.preventDefault(); drop.classList.remove("on"); if (ready && e.dataTransfer.files[0]) load(e.dataTransfer.files[0]); };
$("#file").onchange = e => { if (e.target.files[0]) load(e.target.files[0]); };

async function load(file) {
  if (!ready) return;
  try {
    fileName = file.name;
    drop.classList.add("lock");
    const myBeat = ++beat;
    status("Декодирую " + file.name + "…", false, false, 4);
    const buf = await file.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const ab = await ctx.decodeAudioData(buf);
    if (ctx.close) ctx.close();
    sr = ab.sampleRate;
    const L = ab.getChannelData(0);
    const R = ab.numberOfChannels > 1 ? ab.getChannelData(1) : L;
    nSamples = ab.length;
    const interleaved = new Float32Array(nSamples * 2);
    interleaved.set(L, 0); interleaved.set(R, nSamples);
    const keep = interleaved.slice();
    audioData = { copy: keep, duration: ab.duration };
    let stems = $("#stemsOn").checked;
    const stemsTooLong = stems && ab.duration > 150;
    if (stemsTooLong) { stems = false; $("#stemsOn").checked = false; }
    const iters = nSamples > 44100 * 90 ? 4 : 6;
    status("Мастерю «" + file.name + "» " + Math.round(ab.duration) + "s — страница живая, работаю в фоне" + (stemsTooLong ? " (дорожки отключил: трек длинный, не влезет в память телефона)" : "") + "…", false, false, 8);
    worker.postMessage({ t: "load", buf: interleaved.buffer, n: nSamples, sr, stems, iters }, [interleaved.buffer]);
    const w = setInterval(() => {
      if (beat !== myBeat) { clearInterval(w); return; }
      const v = parseFloat($("#pbarfill").style.width) || 12;
      if (v < 92) $("#pbarfill").style.width = (v + 1.5) + "%";
    }, 700);
  } catch (e) {
    status("Ошибка: " + String(e.message || e).split("\n").slice(-4).join(" | "), true);
  } finally {
    drop.classList.remove("lock");
  }
}

let urlCache = {};
const blobOf = p => blobs[p] ? (urlCache[p] ||= URL.createObjectURL(new Blob([blobs[p]], { type: "audio/wav" }))) : "";

function fmt(v) { return typeof v === "number" ? (Math.abs(v) > 1000 ? v.toFixed(0) : v.toFixed(2)) : v; }

function show() {
  $("#results").style.display = "block";
  const mt = $("#metrics"); mt.innerHTML = "<tr><th></th><th>До</th><th>После</th></tr>";
  const M = [["lufs", "LUFS (громкость)"], ["peak_dbtp", "True Peak, dBTP"], ["centroid_hz", "Спектр. центр, Hz"], ["stereo_width_db", "Стерео-ширина, dB"], ["crest_db", "Крест-фактор, dB"]];
  for (const [k, l] of M) {
    const a = lastRes.before[k] ?? 0, b = lastRes.after[k] ?? 0;
    mt.insertAdjacentHTML("beforeend", `<tr><td>${l}</td><td>${fmt(a)}</td><td class="${b > a ? 'good' : 'warn'}">${fmt(b)}</td></tr>`);
  }
  $("#meters").innerHTML = `<div class="meter"><div class="v">${lastRes.after.lufs.toFixed(1)}</div><div class="l">LUFS</div></div>
    <div class="meter"><div class="v">${lastRes.after.peak_dbtp.toFixed(1)}</div><div class="l">dBTP</div></div>
    <div class="meter"><div class="v">${lastRes.loss.toFixed(2)}</div><div class="l">Loss</div></div>
    <div class="meter"><div class="v">${lastRes.pre_gain_db > 0 ? '+' : ''}${lastRes.pre_gain_db.toFixed(1)}</div><div class="l">Пре-трим, dB</div></div>`;

  $("#au0").src = blobOf("/session/original.wav");
  $("#au1").src = blobOf("/session/mastered.wav");
  $("#waves").innerHTML = "";
  drawWave(blobs["/session/original.wav"], false);
  drawWave(blobs["/session/mastered.wav"], true);
  buildParams(lastRes.params);
  drawStems();
  downloads();
  setTimeout(spectrum, 100);
  $("#stemsCard").style.display = lastRes.stems ? "block" : "none";
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildParams(p) {
  const box = $("#params"); box.innerHTML = "";
  addSl(box, "Громкость (LUFS)", p.target_lufs, -22, -8, 0.5, v => p.target_lufs = v);
  addSl(box, "Потолок (dBTP)", p.ceiling_dbtp, -2.5, -0.2, 0.1, v => p.ceiling_dbtp = v);
  addSl(box, "Стерео-ширина", p.stereo_width, 0.6, 1.7, 0.05, v => p.stereo_width = v);
  addSl(box, "HPF, Hz", p.hpf_hz, 16, 80, 1, v => p.hpf_hz = v);
  addSl(box, "Сатурация", p.saturation, 0, 0.8, 0.02, v => p.saturation = v);
  addSl(box, "Эксайтер", p.exciter.mix, 0, 0.35, 0.01, v => p.exciter.mix = v);
  addSl(box, "Де-эссер", p.de_ess.amount, 0, 0.8, 0.05, v => p.de_ess.amount = v);
  const h = document.createElement("div");
  h.textContent = "EQ по полосам (dB):"; h.style.cssText = "color:var(--dim);margin:12px 0 4px;font-size:12px";
  box.appendChild(h);
  for (const k of Object.keys(p.band_gains_db).sort((a, b) => parseFloat(a) - parseFloat(b)))
    addSl(box, k + " Hz", p.band_gains_db[k], -6, 6, 0.25, v => p.band_gains_db[k] = v);
}
function addSl(box, label, val, min, max, step, set) {
  const l = document.createElement("label"); l.innerHTML = `<span>${label}</span>`;
  const i = document.createElement("input"); i.type = "range"; i.min = min; i.max = max; i.step = step; i.value = val;
  const s = document.createElement("span"); s.textContent = (+val).toFixed(2); s.style.cssText = "min-width:52px;text-align:right";
  i.oninput = () => { s.textContent = (+i.value).toFixed(2); set(+i.value); };
  l.append(i, s); box.appendChild(l);
}

$("#rerun").onclick = () => {
  if (!audioData) return;
  status("Пересчитываю с твоими настройками…", false, false, 10);
  const snd = audioData.copy.slice(0, 2 * nSamples);
  worker.postMessage({ t: "load", buf: snd.buffer, n: nSamples, sr, stems: false, iters: 5, params: lastRes ? lastRes.params : null }, [snd.buffer]);
};

$("#ab").onclick = () => {
  const a = $("#au0"), b = $("#au1"), t = a.currentTime || b.currentTime || 0;
  a.pause(); b.pause();
  (isA ? a : b).currentTime = t;
  (isA ? a : b).play();
  isA = !isA;
};
let isA = false;

const N2RU = { drums: "🥁 Барабаны", bass: "🎸 Бас", vocals: "🎤 Вокал", other: "🎹 Остальное" };
function drawStems() {
  const L = $("#stemsList"); if (!lastRes.stems) { L.innerHTML = ""; return; }
  L.innerHTML = "";
  for (const [k, v] of Object.entries(lastRes.stems)) {
    const d = document.createElement("div"); d.className = "stem";
    d.innerHTML = `<h3>${N2RU[k] || k}</h3>
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        <div class="pl"><h4><span>Сырой стем</span></h4><audio controls src="${blobOf("/session/stem_" + k + ".wav")}"></audio></div>
        <div class="pl"><h4><span>Мастер</span></h4><audio controls src="${blobOf("/session/stemm_" + k + ".wav")}"></audio></div>
      </div>
      <small>LUFS ${v.before.lufs.toFixed(1)} → <b class="good">${v.after.lufs.toFixed(1)}</b> · TP ${v.before.peak_dbtp.toFixed(1)} → ${v.after.peak_dbtp.toFixed(1)} dBTP</small>`;
    L.appendChild(d);
  }
}

function downloads() {
  const d = $("#dls"); d.innerHTML = "";
  const items = [["Мастер", "/session/mastered.wav"], ["Оригинал", "/session/original.wav"]];
  if (lastRes.stems) items.unshift(["Стем-микс", "/session/stem_mix.wav"]);
  for (const [name, p] of items)
    d.insertAdjacentHTML("beforeend", `<a class="dl" href="${blobOf(p)}" download="${name.replace(/ /g, "_")}.wav">⬇ ${name}</a>`);
  for (const k of Object.keys(lastRes.stems || {}))
    d.insertAdjacentHTML("beforeend", `<a class="dl" href="${blobOf("/session/stemm_" + k + ".wav")}" download="stem_${k}_master.wav">⬇ ${N2RU[k]}</a>`);
}

function pcmMono(u8) {
  const dv = new DataView(u8.buffer ? u8.buffer : u8, u8.byteOffset || 0, u8.byteLength);
  const dataOff = 44;
  const nSmp = Math.floor((u8.byteLength - dataOff) / 4);
  const data = new Float32Array(Math.max(0, nSmp));
  for (let i = 0; i < nSmp; i++) data[i] = dv.getInt16(dataOff + i * 4, true) / 32767;
  return { data, sr: dv.getUint32(24, true) };
}

function drawWave(u8, isMaster) {
  const dv = new DataView(u8.buffer ? u8.buffer : u8, u8.byteOffset || 0, u8.byteLength);
  const dataOff = 44;
  const nSmp = Math.floor((u8.byteLength - dataOff) / 4);
  if (nSmp <= 4) return;
  const n = 600, blk = Math.max(1, Math.floor(nSmp / n));
  const pk = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let m = 0; const off = dataOff + i * blk * 4;
    for (let j = 0; j < blk && off + j * 4 + 3 < u8.byteLength; j++) {
      const v = (Math.abs(dv.getInt16(off + j * 4, true)) + Math.abs(dv.getInt16(off + j * 4 + 2, true))) / 32767;
      if (v > m) m = v;
    }
    pk[i] = m;
  }
  const c = document.createElement("canvas");
  c.width = 1000; c.height = 44; c.style.height = "44px"; c.style.marginBottom = "8px";
  c.title = isMaster ? "Мастер" : "Оригинал";
  const g = c.getContext("2d");
  g.fillStyle = isMaster ? "rgba(63,224,176,.85)" : "rgba(85,115,190,.8)";
  pk.forEach((v, i) => { const h = Math.min(1, v) * 42; g.fillRect(i * (1000 / n), (44 - h) / 2, 1000 / n, h); });
  $("#waves").appendChild(c);
}

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j2 = 0; j2 < len / 2; j2++) {
        const ur = re[i + j2], ui = im[i + j2];
        const xr = re[i + j2 + len / 2], xi = im[i + j2 + len / 2];
        const vr = xr * cr - xi * ci, vi = xr * ci + xi * cr;
        re[i + j2] = ur + vr; im[i + j2] = ui + vi;
        re[i + j2 + len / 2] = ur - vr; im[i + j2 + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

async function spectrum() {
  try {
    const w0 = pcmMono(blobs["/session/original.wav"]), w1 = pcmMono(blobs["/session/mastered.wav"]);
    const x0 = w0.data, x1 = w1.data, srate = w0.sr || 44100;
    const N = 4096, frames = 20, hop = Math.max(N, Math.floor(Math.min(x0.length, x1.length) / frames));
    const w = new Float32Array(N); for (let i = 0; i < N; i++) w[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (N - 1));
    const spec = data => {
      const out = new Float32Array(56).fill(-120);
      for (let k = 0; k < frames; k++) {
        const base = Math.min(k * hop, data.length - N);
        if (base < 0) break;
        const re = new Float32Array(N), im = new Float32Array(N);
        for (let i = 0; i < N; i++) re[i] = data[base + i] * w[i];
        fft(re, im);
        for (let i = 1; i < N / 2; i++) {
          const f = i * srate / N;
          const b = Math.max(0, Math.min(55, Math.floor(Math.log2(f / 31) * 10)));
          const m = 20 * Math.log10(Math.hypot(re[i], im[i]) / (N / 4) + 1e-9);
          if (m > out[b]) out[b] = m;
        }
      }
      return out;
    };
    const A = spec(x0), B = spec(x1);
    const c = $("#spec"), g = c.getContext("2d");
    g.fillStyle = "#0a0e18"; g.fillRect(0, 0, c.width, c.height);
    const W = c.width / A.length;
    for (let i = 0; i < A.length; i++) {
      const hA = Math.max(3, Math.min(c.height, (A[i] + 100) / 100 * c.height));
      const hB = Math.max(3, Math.min(c.height, (B[i] + 100) / 100 * c.height));
      g.fillStyle = "#38507a"; g.fillRect(i * W + 1, c.height - hA, W - 2, hA);
      g.fillStyle = "#3fe0b0"; g.fillRect(i * W + 1, c.height - hB, W - 2, hB);
    }
  } catch (e) { /* спектр — косметика */ }
}
