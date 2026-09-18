const state = {
  latentDim: 20,
  muMin: -5,
  muMax: 5,
  sigmaMin: 0,
  sigmaMax: 3,
  mu: [],
  sigma: [],
  eps: [],
  mode: "mean",
  original: null,
  label: null,
  index: null,
  morphFrom: null,
  morphTo: null,
  catalogOffset: 0,
};

let decodeTimer = null;
let samplesTimer = null;
let sweepToken = 0;

const $ = (id) => document.getElementById(id);

function zeros(n, v = 0) {
  return Array.from({ length: n }, () => v);
}

function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function paint(canvas, b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const ctx = canvas.getContext("2d");
  const im = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = bytes[i];
    const o = i * 4;
    im.data[o] = v;
    im.data[o + 1] = v;
    im.data[o + 2] = v;
    im.data[o + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

function dimKl(i) {
  const mu = state.mu[i];
  const s = Math.max(state.sigma[i], 1e-8);
  const logvar = Math.log(s * s);
  return -0.5 * (1 + logvar - mu * mu - s * s);
}

function fmt(x) {
  return Number(x).toFixed(2);
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || err));
  }
  return res.json();
}

function payload() {
  return {
    mu: state.mu,
    sigma: state.sigma,
    mode: state.mode,
    eps: state.eps,
  };
}

function renderRows() {
  const root = $("rows");
  root.innerHTML = "";
  for (let i = 0; i < state.latentDim; i++) {
    const row = document.createElement("div");
    row.className = "row";
    row.dataset.i = String(i);
    row.innerHTML = `
      <button type="button" class="idx" title="reset this dimension">z${i}</button>
      <span class="lbl">μ</span>
      <input type="range" class="mu-range" min="${state.muMin}" max="${state.muMax}" step="0.01">
      <input type="number" class="num mu-num" step="0.01">
      <span class="lbl sigma-lbl">σ</span>
      <input type="range" class="sigma-range" min="${state.sigmaMin}" max="${state.sigmaMax}" step="0.01">
      <input type="number" class="num sigma-num" step="0.01">
      <span class="var"></span>
      <button type="button" class="sweep">sweep</button>
    `;
    root.appendChild(row);
    bindRow(row, i);
    syncRow(i);
  }
}

function bindRow(row, i) {
  row.querySelector(".mu-range").addEventListener("input", (e) => {
    state.mu[i] = Number(e.target.value);
    syncRow(i);
    scheduleDecode();
  });
  row.querySelector(".mu-num").addEventListener("change", (e) => {
    state.mu[i] = Number(e.target.value);
    syncRow(i);
    scheduleDecode();
  });
  row.querySelector(".sigma-range").addEventListener("input", (e) => {
    state.sigma[i] = Math.max(0, Number(e.target.value));
    syncRow(i);
    scheduleDecode();
  });
  row.querySelector(".sigma-num").addEventListener("change", (e) => {
    state.sigma[i] = Math.max(0, Number(e.target.value));
    syncRow(i);
    scheduleDecode();
  });
  row.querySelector(".idx").addEventListener("click", () => {
    state.mu[i] = 0;
    state.sigma[i] = 1;
    syncRow(i);
    scheduleDecode();
  });
  row.querySelector(".sweep").addEventListener("click", () => sweepDim(i));
}

function syncRow(i) {
  const row = $("rows").children[i];
  if (!row) return;
  row.querySelector(".mu-range").value = state.mu[i];
  row.querySelector(".mu-num").value = fmt(state.mu[i]);
  row.querySelector(".sigma-range").value = state.sigma[i];
  row.querySelector(".sigma-num").value = fmt(state.sigma[i]);
  row.querySelector(".var").textContent = fmt(state.sigma[i] * state.sigma[i]);
  row.querySelector(".idx").classList.toggle("hot", dimKl(i) > 0.15);
}

function syncAllRows() {
  for (let i = 0; i < state.latentDim; i++) syncRow(i);
}

function scheduleDecode() {
  clearTimeout(decodeTimer);
  decodeTimer = setTimeout(decode, 40);
  clearTimeout(samplesTimer);
  samplesTimer = setTimeout(drawSamples, 180);
}

async function decode() {
  try {
    const out = await api("/api/decode", payload());
    paint($("output"), out.pixels);
    state.eps = out.eps;
    const src =
      state.label == null ? "prior / manual" : `digit ${state.label}  #${state.index}`;
    $("stats").textContent = `KL(q‖p) ${out.kl.toFixed(2)} nats · ${src} · ${
      state.mode === "mean" ? "z = μ" : "z = μ + σ·ε"
    }`;
    $("stats").classList.remove("error");
    $("out-caption").textContent = state.mode === "mean" ? "decode μ" : "decode sample";
  } catch (err) {
    $("stats").textContent = String(err.message || err);
    $("stats").classList.add("error");
  }
}

async function drawSamples() {
  const grid = $("sample-grid");
  try {
    const out = await api("/api/samples", { ...payload(), n: 8 });
    grid.innerHTML = "";
    for (const pixels of out.images) {
      const c = document.createElement("canvas");
      c.width = 28;
      c.height = 28;
      paint(c, pixels);
      grid.appendChild(c);
    }
  } catch {
    /* keep the last grid if a in-flight slider request fails */
  }
}

function showOriginal(b64) {
  $("orig-wrap").hidden = !b64;
  if (b64) paint($("original"), b64);
}

async function encodeBody(body) {
  const out = await api("/api/encode", body);
  state.mu = out.mu;
  state.sigma = out.sigma;
  state.eps = zeros(state.latentDim);
  state.original = out.original;
  state.label = out.label;
  state.index = out.index;
  state.morphFrom = null;
  state.morphTo = null;
  $("morph-t").value = "0";
  $("morph-t").disabled = true;
  $("morph-t-val").textContent = "0.00";
  $("morph-digit").value = "";
  showOriginal(out.original);
  syncAllRows();
  markDigit(out.label);
  markFilm(out.index);
  await decode();
  await drawSamples();
}

function markDigit(label) {
  for (const btn of $("digits").querySelectorAll("button")) {
    btn.classList.toggle("active", Number(btn.dataset.digit) === label);
  }
}

function markFilm(index) {
  for (const btn of $("filmstrip").querySelectorAll(".thumb")) {
    btn.classList.toggle("active", Number(btn.dataset.index) === index);
  }
}

function resetPriorParams(sampleEps = true) {
  state.mu = zeros(state.latentDim, 0);
  state.sigma = zeros(state.latentDim, 1);
  state.eps = sampleEps
    ? Array.from({ length: state.latentDim }, randn)
    : zeros(state.latentDim);
  state.original = null;
  state.label = null;
  state.index = null;
  state.morphFrom = null;
  state.morphTo = null;
  $("morph-t").disabled = true;
  $("morph-digit").value = "";
  showOriginal(null);
  markDigit(null);
  markFilm(null);
  syncAllRows();
}

async function samplePrior() {
  resetPriorParams(true);
  state.mode = "sample";
  document.querySelector('input[name="mode"][value="sample"]').checked = true;
  await decode();
  await drawSamples();
}

async function resetStandard() {
  resetPriorParams(false);
  state.mode = "mean";
  document.querySelector('input[name="mode"][value="mean"]').checked = true;
  await decode();
  await drawSamples();
}

async function resampleEps() {
  state.eps = Array.from({ length: state.latentDim }, randn);
  state.mode = "sample";
  document.querySelector('input[name="mode"][value="sample"]').checked = true;
  await decode();
  await drawSamples();
}

async function nudge() {
  state.mu = state.mu.map((m) => m + 0.25 * randn());
  syncAllRows();
  await decode();
  await drawSamples();
}

async function sweepDim(i) {
  const token = ++sweepToken;
  const saved = state.mu[i];
  for (let x = state.muMin; x <= state.muMax + 1e-9; x += 0.12) {
    if (token !== sweepToken) return;
    state.mu[i] = x;
    syncRow(i);
    await decode();
    await new Promise((r) => setTimeout(r, 16));
  }
  if (token !== sweepToken) return;
  state.mu[i] = saved;
  syncRow(i);
  await decode();
}

async function startMorph(digit) {
  if (digit === "") {
    state.morphFrom = null;
    state.morphTo = null;
    $("morph-t").disabled = true;
    return;
  }
  const out = await api("/api/encode", { digit: Number(digit) });
  state.morphFrom = { mu: state.mu.slice(), sigma: state.sigma.slice() };
  state.morphTo = { mu: out.mu, sigma: out.sigma, label: out.label, index: out.index };
  $("morph-t").disabled = false;
  $("morph-t").value = "0";
  $("morph-t-val").textContent = "0.00";
}

function applyMorph(t) {
  if (!state.morphFrom || !state.morphTo) return;
  const a = state.morphFrom;
  const b = state.morphTo;
  state.mu = a.mu.map((v, i) => (1 - t) * v + t * b.mu[i]);
  state.sigma = a.sigma.map((v, i) => (1 - t) * v + t * b.sigma[i]);
  $("morph-t-val").textContent = t.toFixed(2);
  syncAllRows();
  scheduleDecode();
}

async function loadCatalog(append = false) {
  if (!append) state.catalogOffset = 0;
  const res = await fetch(`/api/catalog?n=32&offset=${state.catalogOffset}`);
  const data = await res.json();
  const strip = $("filmstrip");
  if (!append) strip.innerHTML = "";
  for (const item of data.items) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb";
    btn.dataset.index = String(item.index);
    btn.title = `#${item.index} digit ${item.label}`;
    const c = document.createElement("canvas");
    c.width = 28;
    c.height = 28;
    paint(c, item.pixels);
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = String(item.label);
    btn.append(c, tag);
    btn.addEventListener("click", () => encodeBody({ index: item.index }));
    strip.appendChild(btn);
  }
  state.catalogOffset += data.items.length;
  if (state.index != null) markFilm(state.index);
}

function buildDigitButtons() {
  const box = $("digits");
  const morph = $("morph-digit");
  for (let d = 0; d <= 9; d++) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.dataset.digit = String(d);
    btn.textContent = String(d);
    btn.title = `encode a random test ${d}`;
    btn.addEventListener("click", () => encodeBody({ digit: d }));
    box.appendChild(btn);
    const opt = document.createElement("option");
    opt.value = String(d);
    opt.textContent = String(d);
    morph.appendChild(opt);
  }
}

function bindControls() {
  for (const el of document.querySelectorAll('input[name="mode"]')) {
    el.addEventListener("change", () => {
      state.mode = el.value;
      scheduleDecode();
    });
  }
  $("btn-prior").addEventListener("click", samplePrior);
  $("btn-reset").addEventListener("click", resetStandard);
  $("btn-eps").addEventListener("click", resampleEps);
  $("btn-nudge").addEventListener("click", nudge);
  $("btn-random").addEventListener("click", () => encodeBody({}));
  $("btn-more").addEventListener("click", () => loadCatalog(true));
  $("btn-redraw").addEventListener("click", drawSamples);
  $("morph-digit").addEventListener("change", (e) => startMorph(e.target.value));
  $("morph-t").addEventListener("input", (e) => applyMorph(Number(e.target.value)));
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key >= "0" && e.key <= "9") encodeBody({ digit: Number(e.key) });
    else if (e.key === "r" || e.key === "R") encodeBody({});
    else if (e.key === "p" || e.key === "P") samplePrior();
    else if (e.key === "e" || e.key === "E") resampleEps();
  });
}

async function init() {
  const info = await fetch("/api/info").then((r) => r.json());
  state.latentDim = info.latent_dim;
  state.muMin = info.mu_min;
  state.muMax = info.mu_max;
  state.sigmaMin = info.sigma_min;
  state.sigmaMax = info.sigma_max;
  $("meta").innerHTML = `${info.arch} · ${info.latent_dim}-D · epoch ${info.epoch}<br>${info.device} · ${info.checkpoint}`;
  state.mu = zeros(info.latent_dim, 0);
  state.sigma = zeros(info.latent_dim, 1);
  state.eps = zeros(info.latent_dim, 0);
  buildDigitButtons();
  bindControls();
  renderRows();
  await loadCatalog(false);
  await encodeBody({});
}

init().catch((err) => {
  $("meta").textContent = String(err);
  $("meta").classList.add("error");
});
