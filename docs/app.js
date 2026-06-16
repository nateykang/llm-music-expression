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
    renderScore(); // in case a piece was already selected
  });

  let batches = [];
  try {
    const idx = await fetchJSON("data/index.json");
    batches = idx.batches || [];
  } catch (e) {
    setStatus("No batches found yet. Run the CLI to generate some (see README).");
    return;
  }
  if (!batches.length) {
    setStatus("No batches found yet. Run the CLI to generate some.");
    return;
  }
  fillSelect(els.batch, batches);
  els.batch.onchange = loadBatch;
  els.prompt.onchange = async () => { refreshModels(); await onSelectChange(); };
  els.model.onchange = onSelectChange;
  await loadBatch();
}

async function loadBatch() {
  const dir = `data/${els.batch.value}`;
  manifest = await fetchJSON(`${dir}/data.json`);
  manifest._dir = dir;
  fillSelect(els.prompt, unique(manifest.pieces.map((p) => p.prompt)));
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

  await renderScore();
}

async function renderScore() {
  const piece = current();
  if (!piece) return;
  if (!piece.ok || !piece.score) {
    els.score.innerHTML = `<p class="note">${piece.error ? "Generation failed: " + piece.error : "No score available."}</p>`;
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
    els.score.innerHTML = svg;
  } catch (e) {
    els.score.innerHTML = `<p class="note">Could not engrave score: ${e}</p>`;
  }
}

// --- helpers ------------------------------------------------------------------
function current() {
  if (!manifest) return null;
  return manifest.pieces.find(
    (p) => p.prompt === els.prompt.value && p.model === els.model.value
  );
}
function fillSelect(sel, items) {
  const prev = sel.value;
  sel.innerHTML = "";
  for (const it of items) {
    const o = document.createElement("option");
    o.value = o.textContent = it;
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
