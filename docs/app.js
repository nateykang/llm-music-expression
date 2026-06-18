// Static viewer: load baked batches, engrave MusicXML live with Verovio, play audio.

const els = {
  batch: document.getElementById("batch"),
  prompt: document.getElementById("prompt"),
  model: document.getElementById("model"),
  title: document.getElementById("title"),
  short: document.getElementById("short"),
  long: document.getElementById("long"),
  audioSlot: document.getElementById("audio-slot"),
  score: document.getElementById("score"),
  status: document.getElementById("status"),
  modelLabel: document.getElementById("model-label"),
  compare: document.getElementById("compare"),
  single: document.getElementById("single"),
  grid: document.getElementById("compare-grid"),
  mode: document.getElementById("mode"),
  modeLabel: document.getElementById("mode-label"),
  compareGen: document.getElementById("compare-gen"),
  compareGenLabel: document.getElementById("compare-gen-label"),
  promptPanel: document.getElementById("prompt-panel"),
  promptMode: document.getElementById("prompt-mode"),
  sysPrompt: document.getElementById("sys-prompt"),
  userPrompt: document.getElementById("user-prompt"),
};

let tk = null; // Verovio toolkit
let manifest = null; // current batch data.json

// --- Verovio init (WASM loads asynchronously) ---------------------------------
const verovioReady = new Promise((resolve) => {
  if (window.verovio && verovio.module) {
    verovio.module.onRuntimeInitialized = () => resolve(new verovio.toolkit());
  } else {
    // Script may still be loading; poll briefly.
    const iv = setInterval(() => {
      if (window.verovio && verovio.module) {
        clearInterval(iv);
        verovio.module.onRuntimeInitialized = () => resolve(new verovio.toolkit());
      }
    }, 50);
  }
});

async function init() {
  // Only one piece audible at a time: when any <audio> starts, pause the others
  // (matters most in compare mode, which renders one player per model).
  document.addEventListener(
    "play",
    (e) => {
      for (const a of document.querySelectorAll("audio")) {
        if (a !== e.target) a.pause();
      }
    },
    true
  );

  verovioReady.then((toolkit) => {
    tk = toolkit;
    tk.setOptions({ pageWidth: 1800, scale: 40, adjustPageHeight: true, footer: "none", header: "none" });
    onSelectChange(); // in case a piece was already selected before the engraver loaded
  });

  let batches = [];
  try {
    const idx = await fetchJSON(`data/index.json?t=${Date.now()}`);
    batches = idx.batches || [];
  } catch (e) {
    setStatus("No batches found yet. Run the CLI to generate some (see README).");
    return;
  }
  if (!batches.length) {
    setStatus("No batches found yet. Run the CLI to generate some.");
    return;
  }
  fillSelect(els.batch, batches, batchLabel);
  // Newest batch (index.json is sorted newest-first) is the canonical one and
  // loads by default. Reveal the picker only if explicitly opted in.
  if (document.body.dataset.showBatch === "1") {
    document.getElementById("batch-label").hidden = false;
  }

  // Discover each batch's generation mode (from its first piece) so the user can
  // flip between e.g. code-gen and ABC versions of the same model×prompt grid.
  const metas = await Promise.all(
    batches.map((b) =>
      fetchJSON(`data/${b}/data.json`).then((m) => ({ dir: b, mode: m.pieces?.[0]?.mode })).catch(() => null)
    )
  );
  modeToBatch = {};
  for (const m of metas) {
    if (m && m.mode && !(m.mode in modeToBatch)) modeToBatch[m.mode] = m.dir; // batches are newest-first
  }
  const modes = Object.keys(modeToBatch);
  if (modes.length > 1) {
    fillSelect(els.mode, modes, modeLabel);
    els.mode.value = modes.includes("codegen") ? "codegen" : modes[0];
    els.batch.value = modeToBatch[els.mode.value];
    els.modeLabel.hidden = false;
    els.mode.onchange = () => { els.batch.value = modeToBatch[els.mode.value]; loadBatch(); };
    els.compareGenLabel.hidden = false; // comparing methods only makes sense with 2+
  }

  els.batch.onchange = loadBatch;
  els.prompt.onchange = async () => { refreshModels(); await onSelectChange(); };
  els.model.onchange = onSelectChange;
  // The two compare views are mutually exclusive.
  els.compare.onchange = () => { if (els.compare.checked) els.compareGen.checked = false; onSelectChange(); };
  els.compareGen.onchange = () => { if (els.compareGen.checked) els.compare.checked = false; onSelectChange(); };
  await loadBatch();
}

// Friendly labels for the generation-mode toggle.
function modeLabel(mode) {
  return { codegen: "Code (music21)", abc: "ABC notation" }[mode] || mode;
}
let modeToBatch = {};

const _manifests = {}; // dir -> manifest (cached; the other generation method needs it too)
async function getManifest(dir) {
  if (!_manifests[dir]) {
    const m = await fetchJSON(`${dir}/data.json`);
    m._dir = dir;
    m._labels = {};
    for (const p of m.pieces) m._labels[p.prompt] = p.prompt_label || p.prompt;
    _manifests[dir] = m;
  }
  return _manifests[dir];
}

async function loadBatch() {
  manifest = await getManifest(`data/${els.batch.value}`);
  fillSelect(els.prompt, unique(manifest.pieces.map((p) => p.prompt)), (id) => manifest._labels[id]);
  refreshModels();
  await onSelectChange();
}

function refreshModels() {
  const models = unique(
    manifest.pieces.filter((p) => p.prompt === els.prompt.value).map((p) => p.model)
  );
  fillSelect(els.model, models);
}

async function onSelectChange() {
  updatePromptPanel();
  const byModel = els.compare.checked;       // compare models (fix prompt+method)
  const byMethod = els.compareGen.checked;    // compare methods (fix prompt+model)
  const grid = byModel || byMethod;
  // In compare-models the model selector is irrelevant; in compare-methods the
  // generation selector is (we show every method); single view shows both.
  els.modelLabel.hidden = byModel;
  els.modeLabel.hidden = byMethod || Object.keys(modeToBatch).length < 2;
  els.single.hidden = grid;
  els.grid.hidden = !grid;
  if (byMethod) await renderCompareMethods();
  else if (byModel) await renderCompare();
  else await renderSingle();
}

async function renderSingle() {
  const piece = current();
  if (!piece) return;
  els.title.textContent = piece.title || "Untitled";
  els.short.textContent = piece.short_description || "";
  els.long.textContent = piece.long_description || "";
  await mountMedia(els.score, els.audioSlot, piece, manifest._dir);
}

// One column per model for the currently selected prompt + generation method.
async function renderCompare() {
  const pieces = manifest.pieces.filter((p) => p.prompt === els.prompt.value);
  els.grid.innerHTML = "";
  for (const piece of pieces) {
    await addCompareCard(piece, manifest._dir, piece.model);
  }
}

// One column per generation method (code-gen / ABC) for the current model + prompt.
async function renderCompareMethods() {
  els.grid.innerHTML = "";
  for (const mode of Object.keys(modeToBatch)) {
    const m = await getManifest(`data/${modeToBatch[mode]}`);
    const piece = m.pieces.find(
      (p) => p.prompt === els.prompt.value && p.model === els.model.value
    );
    if (!piece) continue;
    await addCompareCard(piece, m._dir, modeLabel(mode));
  }
}

// Build one comparison card (shared by both compare views).
async function addCompareCard(piece, dir, header) {
  const card = document.createElement("article");
  card.className = "compare-card";
  card.innerHTML = `
    <h3 class="model-name">${header}</h3>
    <p class="piece-title">${piece.ok ? (piece.title || "Untitled") : "—"}</p>
    <p class="short">${piece.short_description || ""}</p>
    <div class="audio-slot"></div>
    <details><summary>Model's reflection</summary><p>${piece.long_description || ""}</p></details>
    <div class="compare-score"></div>`;
  els.grid.appendChild(card);
  await mountMedia(card.querySelector(".compare-score"), card.querySelector(".audio-slot"), piece, dir);
}

// Mount notation + audio for a piece, picking the engine by generation method:
// ABC pieces carry raw ABC (abcjs engraves + plays it); code-gen pieces carry a
// MusicXML score + pre-baked ogg (Verovio + <audio>).
async function mountMedia(scoreEl, audioSlot, piece, dir) {
  const visual = await mountScore(scoreEl, piece, dir);
  mountAudio(audioSlot, piece, dir, visual);
}

async function mountScore(scoreEl, piece, dir) {
  if (!piece.ok) {
    scoreEl.innerHTML = `<p class="note">${piece.error ? "Generation failed: " + piece.error : "No score available."}</p>`;
    return null;
  }
  if (piece.abc) {
    if (!window.ABCJS) { scoreEl.innerHTML = `<p class="note">Loading ABC engraver…</p>`; return null; }
    scoreEl.innerHTML = "";
    try {
      return ABCJS.renderAbc(scoreEl, piece.abc, { responsive: "resize", add_classes: true })[0];
    } catch (e) {
      scoreEl.innerHTML = `<p class="note">Could not render ABC: ${e}</p>`;
      return null;
    }
  }
  await renderScoreInto(scoreEl, piece, dir);
  return null;
}

function mountAudio(slot, piece, dir, visual) {
  slot.innerHTML = "";
  if (piece.abc) {
    if (!visual || !window.ABCJS || !ABCJS.synth.supportsAudio()) {
      slot.innerHTML = `<p class="note">Audio unavailable in this browser.</p>`;
      return;
    }
    const ctrl = document.createElement("div");
    slot.appendChild(ctrl);
    const sc = new ABCJS.synth.SynthController();
    sc.load(ctrl, null, { displayPlay: true, displayProgress: true, displayWarp: false });
    sc.setTune(visual, false, { soundFontUrl: SOUNDFONT }).catch(() => {
      slot.innerHTML = `<p class="note">Could not load audio.</p>`;
    });
    return;
  }
  if (piece.audio) {
    const a = document.createElement("audio");
    a.controls = true;
    a.src = `${dir}/${piece.audio}`;
    slot.appendChild(a);
  } else {
    slot.innerHTML = `<p class="note">No pre-rendered audio.</p>`;
  }
}
const SOUNDFONT = "https://paulrosen.github.io/midi-js-soundfonts/abcjs/";

// Engrave one piece's MusicXML into a target element (code-gen path).
async function renderScoreInto(target, piece, dir) {
  if (!piece.ok || !piece.score) {
    target.innerHTML = `<p class="note">${piece.error ? "Generation failed: " + piece.error : "No score available."}</p>`;
    return;
  }
  if (!tk) {
    setStatus("Loading engraver…");
    return;
  }
  setStatus("");
  try {
    const xml = await (await fetch(`${dir}/${piece.score}`)).text();
    tk.loadData(xml);
    let svg = "";
    const pages = tk.getPageCount();
    for (let i = 1; i <= pages; i++) svg += tk.renderToSVG(i);
    target.innerHTML = svg;
  } catch (e) {
    target.innerHTML = `<p class="note">Could not engrave score: ${e}</p>`;
  }
}

// The prompt text is identical across models for a given prompt+mode, so show it
// once in a shared panel reflecting the currently selected prompt.
function updatePromptPanel() {
  const piece = manifest && manifest.pieces.find((p) => p.prompt === els.prompt.value);
  if (!piece || !piece.prompt_text) {
    els.promptPanel.hidden = true;
    return;
  }
  els.promptPanel.hidden = false;
  els.promptMode.textContent = piece.mode ? `${piece.mode} mode` : "";
  els.sysPrompt.textContent = piece.system_prompt || "(none recorded)";
  els.userPrompt.textContent = piece.prompt_text;
}

// "20260617_131005__models_3_prompts_3" -> "Jun 17, 2026, 1:10 PM · 3 models × 3 prompts"
function batchLabel(dir) {
  const m = dir.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})__models_(\d+)_prompts_(\d+)$/);
  if (!m) return dir;
  const [, y, mo, d, h, mi, , nModels, nPrompts] = m;
  const date = new Date(+y, +mo - 1, +d, +h, +mi);
  const when = date.toLocaleString(undefined, {
    month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
  });
  const plural = (n, w) => `${n} ${w}${n === "1" ? "" : "s"}`;
  return `${when} · ${plural(nModels, "model")} × ${plural(nPrompts, "prompt")}`;
}

// --- helpers ------------------------------------------------------------------
function current() {
  if (!manifest) return null;
  return manifest.pieces.find(
    (p) => p.prompt === els.prompt.value && p.model === els.model.value
  );
}
function fillSelect(sel, items, labelFn) {
  const prev = sel.value;
  sel.innerHTML = "";
  for (const it of items) {
    const o = document.createElement("option");
    o.value = it;
    o.textContent = labelFn ? labelFn(it) : it;
    sel.appendChild(o);
  }
  if (items.includes(prev)) sel.value = prev;
}
function unique(arr) { return [...new Set(arr)]; }
function setStatus(msg) { els.status.textContent = msg; }
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

init();
