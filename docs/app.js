// Static viewer: load every baked batch into one pool, engrave MusicXML live
// with Verovio (code-gen) or ABC with abcjs, play the pre-baked audio.
//
// There is no batch/"experiment" picker: the three selectors (Generation,
// Prompt, Model) cross-filter the pooled pieces — each dropdown lists the
// values available given the other two. When several pieces exist for one
// (prompt, method, model) cell (the sampling runs), a contextual "Piece"
// variant picker appears.

const els = {
  mode: document.getElementById("mode"),
  prompt: document.getElementById("prompt"),
  model: document.getElementById("model"),
  variant: document.getElementById("variant"),
  variantLabel: document.getElementById("variant-label"),
  modeLabel: document.getElementById("mode-label"),
  modelLabel: document.getElementById("model-label"),
  title: document.getElementById("title"),
  short: document.getElementById("short"),
  long: document.getElementById("long"),
  when: document.getElementById("when"),
  audioSlot: document.getElementById("audio-slot"),
  score: document.getElementById("score"),
  status: document.getElementById("status"),
  compare: document.getElementById("compare"),
  single: document.getElementById("single"),
  grid: document.getElementById("compare-grid"),
  compareGen: document.getElementById("compare-gen"),
  promptPanel: document.getElementById("prompt-panel"),
  promptMode: document.getElementById("prompt-mode"),
  sysPrompt: document.getElementById("sys-prompt"),
  userPrompt: document.getElementById("user-prompt"),
};

let tk = null; // Verovio toolkit
let PIECES = []; // every ok piece from every batch, newest batch first
let PROMPT_LABELS = {}; // prompt id -> human label
const SEL = { mode: null, prompt: null, model: null };
const MODE_ORDER = ["codegen", "abc", "smt-abc"];

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

// Only one piece audible at a time, across both audio engines (native <audio>
// for code-gen, abcjs Web Audio synths for ABC). Pause everything except the one
// just started.
function pauseOthers(exceptAudio, exceptSynth) {
  for (const a of document.querySelectorAll("audio")) {
    if (a !== exceptAudio) { try { a.pause(); } catch (e) {} }
  }
  for (const sc of activeSynths) {
    if (sc !== exceptSynth) { try { sc.pause(); } catch (e) {} }
  }
}

// "20260622_195241__models_7_prompts_1" -> Date
function batchDate(dir) {
  const m = dir.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})__/);
  if (!m) return new Date(0);
  const [, y, mo, d, h, mi, s] = m;
  return new Date(+y, +mo - 1, +d, +h, +mi, +s);
}
function fmtWhen(d) {
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

async function init() {
  // When a native <audio> starts, pause every other player (audio + synths).
  document.addEventListener("play", (e) => pauseOthers(e.target, null), true);

  verovioReady.then((toolkit) => {
    tk = toolkit;
    tk.setOptions({ pageWidth: 1800, scale: 40, adjustPageHeight: true, footer: "none", header: "none" });
    if (SEL.mode) onSelectChange(); // re-engrave if a piece was selected before the engraver loaded
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

  // Pool every batch's ok pieces, tagged with their folder + timestamp. Failed
  // generations are dropped — reliability stats live on the Results tab.
  const manifests = await Promise.all(
    batches.map((b) => fetchJSON(`data/${b}/data.json`).then((m) => ({ dir: b, m })).catch(() => null))
  );
  for (const it of manifests.filter(Boolean)) {
    const when = batchDate(it.dir);
    for (const p of it.m.pieces || []) {
      if (!p.ok) continue;
      PROMPT_LABELS[p.prompt] = p.prompt_label || p.prompt;
      PIECES.push(Object.assign({}, p, { _dir: `data/${it.dir}`, _when: when }));
    }
  }
  PIECES.sort((a, b) => (b._when - a._when) || ((a.sample || 0) - (b.sample || 0)));
  if (!PIECES.length) {
    setStatus("No successful pieces found in any batch.");
    return;
  }

  els.mode.onchange = () => refreshSelectors("mode");
  els.prompt.onchange = () => refreshSelectors("prompt");
  els.model.onchange = () => refreshSelectors("model");
  els.variant.onchange = onSelectChange;
  // The two compare views are mutually exclusive.
  els.compare.onchange = () => { if (els.compare.checked) els.compareGen.checked = false; onSelectChange(); };
  els.compareGen.onchange = () => { if (els.compareGen.checked) els.compare.checked = false; onSelectChange(); };

  seedDefaults();
  await refreshSelectors(null);
}

// Friendly labels for the generation-method selector.
function modeLabel(mode) {
  return { codegen: "Code (music21)", abc: "ABC notation", "smt-abc": "SMT-ABC (synchronized)" }[mode] || mode;
}

// --- cross-filtering selectors -------------------------------------------------

function matches(prompt, mode, model) {
  return PIECES.filter(
    (p) =>
      (!prompt || p.prompt === prompt) &&
      (!mode || p.mode === mode) &&
      (!model || p.model === model)
  );
}

// Values available for dimension `k` given the other dimensions in `sel`.
function optionsFor(k, sel) {
  const pool = matches(
    k === "prompt" ? null : sel.prompt,
    k === "mode" ? null : sel.mode,
    k === "model" ? null : sel.model
  );
  const vals = unique(pool.map((p) => (k === "prompt" ? p.prompt : k === "mode" ? p.mode : p.model)));
  if (k === "mode") {
    return MODE_ORDER.filter((m) => vals.includes(m)).concat(vals.filter((v) => !MODE_ORDER.includes(v)));
  }
  if (k === "prompt") {
    return vals.sort((a, b) =>
      a === "free-form" ? -1 : b === "free-form" ? 1
        : (PROMPT_LABELS[a] || a).localeCompare(PROMPT_LABELS[b] || b));
  }
  return vals.sort();
}

function seedDefaults() {
  const modes = optionsFor("mode", SEL);
  SEL.mode = modes.includes("codegen") ? "codegen" : modes[0];
  const prompts = optionsFor("prompt", SEL);
  SEL.prompt = prompts.includes("free-form") ? "free-form" : prompts[0];
  SEL.model = optionsFor("model", SEL)[0];
}

// Rebuild all three dropdowns so each lists the values available given the
// OTHER two selections. When the touched control invalidates the combination,
// re-snap the other two (keeping their old values where still possible).
function refreshSelectors(touched) {
  if (touched && els[touched].value) SEL[touched] = els[touched].value;
  if (touched && !matches(SEL.prompt, SEL.mode, SEL.model).length) {
    const free = ["mode", "prompt", "model"].filter((k) => k !== touched);
    const old = Object.assign({}, SEL);
    for (const k of free) SEL[k] = null;
    for (const k of free) {
      const opts = optionsFor(k, SEL);
      SEL[k] = opts.includes(old[k]) ? old[k] : opts[0];
    }
  }
  const labelFns = { mode: modeLabel, prompt: (id) => PROMPT_LABELS[id] || id, model: null };
  for (const k of ["mode", "prompt", "model"]) {
    const opts = optionsFor(k, Object.assign({}, SEL, { [k]: null }));
    fillSelect(els[k], opts, labelFns[k]);
    els[k].value = SEL[k];
  }
  els.variant.value = "0"; // a new selection starts at its newest piece
  return onSelectChange();
}

// abcjs synths play through Web Audio (not an <audio> element), so they keep
// going when their control UI is cleared. Track them so we can stop them.
let activeSynths = [];
function stopAllMedia() {
  for (const a of document.querySelectorAll("audio")) { try { a.pause(); } catch (e) {} }
  for (const sc of activeSynths) { try { sc.pause(); } catch (e) {} }
  activeSynths = [];
}

async function onSelectChange() {
  stopAllMedia(); // switching pieces/views must stop whatever's currently playing
  const byModel = els.compare.checked;    // compare models (fix prompt+method)
  const byMethod = els.compareGen.checked; // compare methods (fix prompt+model)
  const grid = byModel || byMethod;
  // In compare-models the model selector is irrelevant; in compare-methods the
  // generation selector is (we show every method); single view shows both.
  els.modelLabel.hidden = byModel;
  els.modeLabel.hidden = byMethod;
  els.variantLabel.hidden = true; // renderSingle re-shows it when a cell has variants
  els.single.hidden = grid;
  els.grid.hidden = !grid;
  if (byMethod) await renderCompareMethods();
  else if (byModel) await renderCompare();
  else await renderSingle();
}

async function renderSingle() {
  const list = matches(SEL.prompt, SEL.mode, SEL.model);
  if (!list.length) { setStatus("No piece for this selection."); return; }
  let idx = 0;
  if (list.length > 1) {
    idx = Math.min(parseInt(els.variant.value, 10) || 0, list.length - 1);
    els.variant.innerHTML = "";
    list.forEach((p, i) => {
      const o = document.createElement("option");
      o.value = String(i);
      o.textContent = `${i + 1} of ${list.length} · ${fmtWhen(p._when)}`;
      els.variant.appendChild(o);
    });
    els.variant.value = String(idx);
    els.variantLabel.hidden = false;
  }
  const piece = list[idx];
  updatePromptPanel(piece);
  els.title.textContent = piece.title || "Untitled";
  els.short.textContent = piece.short_description || "";
  els.long.textContent = piece.long_description || "";
  els.when.textContent = `generated ${fmtWhen(piece._when)}`;
  await mountMedia(els.score, els.audioSlot, piece, piece._dir);
}

// One column per model available for the current prompt + generation method.
async function renderCompare() {
  els.grid.innerHTML = "";
  let first = null;
  for (const m of optionsFor("model", Object.assign({}, SEL, { model: null }))) {
    const piece = matches(SEL.prompt, SEL.mode, m)[0];
    if (!piece) continue;
    first = first || piece;
    await addCompareCard(piece, piece._dir, m);
  }
  updatePromptPanel(first);
}

// One column per generation method available for the current prompt + model.
async function renderCompareMethods() {
  els.grid.innerHTML = "";
  let first = null;
  for (const mo of optionsFor("mode", Object.assign({}, SEL, { mode: null }))) {
    const piece = matches(SEL.prompt, mo, SEL.model)[0];
    if (!piece) continue;
    first = first || piece;
    await addCompareCard(piece, piece._dir, modeLabel(mo));
  }
  updatePromptPanel(first);
}

// Build one comparison card (shared by both compare views).
async function addCompareCard(piece, dir, header) {
  const card = document.createElement("article");
  card.className = "compare-card";
  card.innerHTML = `
    <h3 class="model-name">${header}</h3>
    <p class="piece-title">${piece.title || "Untitled"} <span class="note">· ${fmtWhen(piece._when)}</span></p>
    <p class="short">${piece.short_description || ""}</p>
    <div class="audio-slot"></div>
    <details><summary>Model's reflection</summary><p>${piece.long_description || ""}</p></details>
    <div class="compare-score"></div>`;
  els.grid.appendChild(card);
  await mountMedia(card.querySelector(".compare-score"), card.querySelector(".audio-slot"), piece, dir);
}

// Mount notation + audio for a piece, picking the engine by generation method:
// ABC pieces carry raw ABC (abcjs engraves + plays it); code-gen pieces carry a
// MusicXML score + pre-baked audio (Verovio + <audio>).
async function mountMedia(scoreEl, audioSlot, piece, dir) {
  const visual = await mountScore(scoreEl, piece, dir);
  mountAudio(audioSlot, piece, dir, visual);
}

async function mountScore(scoreEl, piece, dir) {
  if (piece.abc) {
    if (!window.ABCJS) { scoreEl.innerHTML = `<p class="note">Loading ABC engraver…</p>`; return null; }
    scoreEl.innerHTML = "";
    let visual = null;
    try {
      // wrap (needs an explicit staffwidth) reflows measures to fit the width.
      // Without it, abcjs honors the model's source line breaks — and models
      // often write an entire voice on one line (e.g. a 41-bar fugue), which
      // responsive-resize then shrinks to unreadably tiny. Wrap to the container
      // width so it re-flows into multiple readable systems instead.
      const staffwidth = Math.max(360, (scoreEl.clientWidth || 740) - 16);
      visual = ABCJS.renderAbc(scoreEl, withInstruments(normalizeAbc(piece.abc)), {
        responsive: "resize",
        add_classes: true,
        staffwidth,
        wrap: { minSpacing: 1.8, maxSpacing: 2.7, preferredMeasuresPerLine: 4 },
      })[0];
    } catch (e) {
      visual = null;
    }
    // abcjs is lenient: malformed ABC yields a blank SVG (0 staves) with no
    // throw. Treat that as a failure and surface it honestly instead of a blank.
    if (!scoreEl.querySelector(".abcjs-staff")) {
      scoreEl.innerHTML =
        `<p class="note">Couldn't engrave this piece — the model's ABC is malformed (invalid syntax abcjs can't parse).</p>` +
        `<details><summary>Show the raw ABC the model wrote</summary><pre class="abc-raw">${escapeHtml(piece.abc)}</pre></details>`;
      return null;
    }
    return visual;
  }
  await renderScoreInto(scoreEl, piece, dir);
  return null;
}

// abcjs plays every voice as piano unless told the instrument. Bind a General
// MIDI program after each named voice header so timbres match the notation.
// Keyword list and order mirror _GM_BY_NAME in src/llm_music/render.py (substring
// match, first hit wins) so the abcjs synth fallback agrees with the baked audio.
const GM_BY_NAME = [
  [/contrabass|double ?bass/, 43], [/violoncello|cello/, 42], [/viola/, 41], [/violin/, 40],
  [/harp/, 46], [/piccolo/, 72], [/flute/, 73], [/oboe/, 68], [/clarinet/, 71], [/bassoon/, 70],
  [/trumpet/, 56], [/trombone/, 57], [/tuba/, 58], [/horn/, 60], [/timpani/, 47],
  [/guitar/, 24], [/organ/, 19], [/harpsichord/, 6], [/sax/, 65], [/piano|keyboard/, 0],
  [/soprano|alto|tenor|bass|choir|voice|vocal/, 52],
];
function gmProgram(name) {
  const n = name.toLowerCase();
  for (const [re, p] of GM_BY_NAME) if (re.test(n)) return p;
  return null;
}
// Mirror the backend's _prepare_abc_for_audio normalization so the engraved
// notation and the baked audio read the same tune:
// 1. abcjs (like abc2midi) treats a blank line as end-of-tune, so a model's
//    mid-tune blank line would render only the first section while the audio
//    (blank-stripped before abc2midi) plays everything. Drop blank lines.
// 2. Some models write inline voice switches as [V1] — but ABC requires [V:V1],
//    and abcjs reads a bare [ as a chord, scrambling the voices. Fix only markers
//    whose id matches a DECLARED voice (V:<id> header), so chords like [CEG] are safe.
function normalizeAbc(abc) {
  let out = abc.split("\n").filter((l) => l.trim()).join("\n");
  const voices = [...new Set([...out.matchAll(/^\s*V:\s*(\S+)/gm)].map((m) => m[1]))];
  for (const v of voices) {
    const esc = v.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    out = out.replace(new RegExp(`\\[${esc}\\]`, "g"), `[V:${v}]`);
  }
  return out;
}

function withInstruments(abc) {
  // If the model wrote its own %%MIDI program lines, respect them everywhere —
  // the backend audio baker skips injection in that case too.
  if (abc.includes("%%MIDI program")) return abc;
  return abc.split("\n").flatMap((line) => {
    // ABC voice names may be quoted (name="Violin I") or a bare token
    // (name=Soprano) — handle both, else we miss the name and default to piano.
    const m = line.match(/^\s*V:\s*\S+.*\bname=(?:"([^"]+)"|(\S+))/);
    if (m) {
      const p = gmProgram(m[1] || m[2]);
      if (p != null) return [line, `%%MIDI program ${p}`];
    }
    return [line];
  }).join("\n");
}
function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function mountAudio(slot, piece, dir, visual) {
  slot.innerHTML = "";
  // ABC audio is pre-baked to MP3 (abc2midi -> FluidSynth), played below like
  // code-gen. Only fall back to the abcjs synth if no MP3 was produced.
  if (piece.abc && !piece.audio) {
    if (!visual || !window.ABCJS || !ABCJS.synth.supportsAudio()) {
      slot.innerHTML = `<p class="note">No audio — the ABC couldn't be parsed.</p>`;
      return;
    }
    const ctrl = document.createElement("div");
    slot.appendChild(ctrl);
    const sc = new ABCJS.synth.SynthController();
    activeSynths.push(sc); // so a later switch can stop it (Web Audio, not <audio>)
    // onStart fires when this synth begins playing -> pause every other player.
    sc.load(ctrl, { onStart: () => pauseOthers(null, sc) }, { displayPlay: true, displayProgress: true, displayWarp: false });
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
  if (!piece.score) {
    target.innerHTML = `<p class="note">No score available.</p>`;
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
// once in a shared panel reflecting the currently shown piece.
function updatePromptPanel(piece) {
  if (!piece || !piece.prompt_text) {
    els.promptPanel.hidden = true;
    return;
  }
  els.promptPanel.hidden = false;
  els.promptMode.textContent = piece.mode ? `${piece.mode} mode` : "";
  els.sysPrompt.textContent = piece.system_prompt || "(none recorded)";
  els.userPrompt.textContent = piece.prompt_text;
}

// --- helpers ------------------------------------------------------------------
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
