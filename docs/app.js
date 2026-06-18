// Static viewer: load baked batches, engrave MusicXML live with Verovio, play audio.

const els = {
  batch: document.getElementById("batch"),
  prompt: document.getElementById("prompt"),
  model: document.getElementById("model"),
  title: document.getElementById("title"),
  short: document.getElementById("short"),
  long: document.getElementById("long"),
  audio: document.getElementById("audio"),
  audioNote: document.getElementById("audio-note"),
  score: document.getElementById("score"),
  status: document.getElementById("status"),
  modelLabel: document.getElementById("model-label"),
  compare: document.getElementById("compare"),
  single: document.getElementById("single"),
  grid: document.getElementById("compare-grid"),
  mode: document.getElementById("mode"),
  modeLabel: document.getElementById("mode-label"),
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
  }

  els.batch.onchange = loadBatch;
  els.prompt.onchange = async () => { refreshModels(); await onSelectChange(); };
  els.model.onchange = onSelectChange;
  els.compare.onchange = onSelectChange;
  await loadBatch();
}

// Friendly labels for the generation-mode toggle.
function modeLabel(mode) {
  return { codegen: "Code (music21)", abc: "ABC notation" }[mode] || mode;
}
let modeToBatch = {};

async function loadBatch() {
  const dir = `data/${els.batch.value}`;
  manifest = await fetchJSON(`${dir}/data.json`);
  manifest._dir = dir;
  // id -> human label (e.g. "free-form" -> "Free form"), from the baked manifest.
  manifest._labels = {};
  for (const p of manifest.pieces) manifest._labels[p.prompt] = p.prompt_label || p.prompt;
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
  const comparing = els.compare.checked;
  els.modelLabel.hidden = comparing;
  els.single.hidden = comparing;
  els.grid.hidden = !comparing;
  if (comparing) {
    await renderCompare();
  } else {
    await renderSingle();
  }
}

async function renderSingle() {
  const piece = current();
  if (!piece) return;

  els.title.textContent = piece.title || "Untitled";
  els.short.textContent = piece.short_description || "";
  els.long.textContent = piece.long_description || "";

  if (piece.audio) {
    els.audio.src = `${manifest._dir}/${piece.audio}`;
    els.audio.hidden = false;
    els.audioNote.hidden = true;
  } else {
    els.audio.removeAttribute("src");
    els.audio.hidden = true;
    els.audioNote.hidden = false;
  }

  await renderScoreInto(els.score, piece);
}

// One column per model for the currently selected prompt, side by side.
async function renderCompare() {
  const pieces = manifest.pieces.filter((p) => p.prompt === els.prompt.value);
  els.grid.innerHTML = "";
  for (const piece of pieces) {
    const card = document.createElement("article");
    card.className = "compare-card";

    const audio = piece.audio
      ? `<audio controls src="${manifest._dir}/${piece.audio}"></audio>`
      : `<p class="note">No pre-rendered audio.</p>`;

    card.innerHTML = `
      <h3 class="model-name">${piece.model}</h3>
      <p class="piece-title">${piece.ok ? (piece.title || "Untitled") : "—"}</p>
      <p class="short">${piece.short_description || ""}</p>
      ${audio}
      <details><summary>Model's reflection</summary><p>${piece.long_description || ""}</p></details>
      <div class="compare-score"></div>`;
    els.grid.appendChild(card);

    await renderScoreInto(card.querySelector(".compare-score"), piece);
  }
}

// Engrave one piece's MusicXML into a target element (shared by both views).
async function renderScoreInto(target, piece) {
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
    const xml = await (await fetch(`${manifest._dir}/${piece.score}`)).text();
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
