/* Composer Studio frontend: login -> session list -> chat with streaming turns.
   Pieces stream in as SSE events; MusicXML engraves via Verovio, ABC via abcjs. */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  models: [],
  defaultModel: "",
  session: null,          // meta of the open session
  pieces: new Map(),      // version (number) -> piece event
  comments: new Map(),    // version (number) -> [comment text] for the open session
  cmpComments: new Map(), // same, for the open comparison's cells
  currentVersion: null,
  streamingChats: new Set(),  // session ids with a turn in flight from this tab
  cmpStreaming: false,
  mode: localStorage.getItem("studio-mode") || "codegen",  // codegen | abc
};

const MODE_LABELS = { codegen: "code", abc: "ABC" };

// ---------- tiny helpers ----------

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* Minimal markdown: paragraphs, **bold**, *italic*, `code`. */
function md(text) {
  return esc(text.trim())
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .split(/\n{2,}/).map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`).join("");
}

function show(view) {
  for (const v of ["login-view", "home-view", "chat-view", "compare-view"])
    $(v).hidden = v !== view;
}

// ---------- login ----------

async function init() {
  try {
    afterLogin(await api("/api/me"));
  } catch (e) {
    show("login-view");
  }
}

$("login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("login-error").hidden = true;
  try {
    await api("/api/login", { method: "POST", body: JSON.stringify({ password: $("password").value }) });
    afterLogin(await api("/api/me"));
  } catch (e) {
    $("login-error").textContent = e.message;
    $("login-error").hidden = false;
  }
});

$("logout").addEventListener("click", async () => {
  await api("/api/logout", { method: "POST" });
  show("login-view");
});

function afterLogin(me) {
  state.models = me.models;
  state.defaultModel = me.default_model;
  const sel = $("new-model");
  sel.innerHTML = "";
  for (const m of me.models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === me.default_model) opt.selected = true;
    sel.appendChild(opt);
  }
  const grid = $("cmp-models");
  grid.innerHTML = "";
  for (const m of me.models) {
    const lbl = document.createElement("label");
    lbl.className = "chk";
    lbl.innerHTML = `<input type="checkbox" value="${m}"> ${esc(m)}`;
    grid.appendChild(lbl);
  }
  showHome();
}

// ---------- home ----------

async function showHome() {
  show("home-view");
  const { sessions } = await api("/api/sessions");
  const list = $("session-list");
  list.innerHTML = "";
  if (!sessions.length) {
    list.innerHTML = '<p class="muted">No sessions yet — start one above.</p>';
    return;
  }
  for (const s of sessions) {
    const isCmp = s.kind === "comparison";
    const btn = document.createElement("button");
    btn.className = "session-item";
    const tag = isCmp ? '<span class="kind">comparison</span>' : "";
    const count = isCmp
      ? `${s.n_versions} result${s.n_versions === 1 ? "" : "s"}`
      : `${esc(s.model)} · ${s.n_versions} version${s.n_versions === 1 ? "" : "s"}`;
    btn.innerHTML =
      `<div class="title">${tag}${esc(s.title)}</div>` +
      `<div class="meta">${new Date(s.last_active * 1000).toLocaleString()} · ${count}</div>`;
    btn.addEventListener("click", () => isCmp ? openComparison(s.id) : openSession(s.id));
    list.appendChild(btn);
  }
}

$("new-session-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const meta = await api("/api/sessions", {
    method: "POST",
    body: JSON.stringify({ title: $("new-title").value, model: $("new-model").value }),
  });
  $("new-title").value = "";
  openSession(meta.id);
});

$("back").addEventListener("click", showHome);

// ---------- chat ----------

async function openSession(id) {
  const { meta, events } = await api(`/api/sessions/${id}`);
  state.session = meta;
  state.pieces = new Map();
  state.comments = new Map();
  state.currentVersion = null;
  $("chat-title").textContent = meta.title;
  $("chat-model").textContent = meta.model;
  const sel = $("turn-model");
  sel.innerHTML = "";
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  }
  sel.value = meta.model;
  $("messages").innerHTML = "";
  $("version-tabs").innerHTML = "";
  $("piece-panel").hidden = true;
  $("piece-empty").hidden = false;
  for (const e of events) {
    if (e.type === "user") addUserBubble(e.text);
    else if (e.type === "assistant") addAssistantHtml(md(e.text));
    else if (e.type === "piece") registerPiece(e, false);
    else if (e.type === "error") addError(e.message);
    else if (e.type === "comment") recordComment(e);
  }
  const last = Math.max(0, ...state.pieces.keys());
  if (last) selectVersion(last);
  // A turn may still be running for THIS session (from this tab or another);
  // reflect that instead of inheriting stale global state.
  const inFlight = state.streamingChats.has(id);
  $("send").disabled = inFlight;
  setStatus(inFlight ? "Still composing — reopen for the result…" : null);
  setPromptEditorOpen(false);
  markPromptState(meta.custom_prompt);
  show("chat-view");
  scrollChat();
  $("input").focus();
}

function recordComment(e) {
  const v = e.version ?? 0;
  if (!state.comments.has(v)) state.comments.set(v, []);
  state.comments.get(v).push(e.text);
}

function scrollChat() {
  const m = $("messages");
  m.scrollTop = m.scrollHeight;
}

function addUserBubble(text) {
  const div = document.createElement("div");
  div.className = "msg user";
  div.textContent = text;
  $("messages").appendChild(div);
  scrollChat();
}

function addAssistantHtml(html) {
  const div = document.createElement("div");
  div.className = "msg assistant";
  div.innerHTML = html;
  $("messages").appendChild(div);
  scrollChat();
  return div;
}

function addError(message) {
  const div = document.createElement("div");
  div.className = "msg error";
  div.textContent = message || "something went wrong";
  $("messages").appendChild(div);
  scrollChat();
}

function addPieceCard(e) {
  const btn = document.createElement("button");
  btn.className = "piece-card";
  btn.innerHTML = `<span class="v">v${e.version}</span>` +
    `<span>${esc(e.title)}</span><span class="note">${esc(e.note || "")}</span>`;
  btn.addEventListener("click", () => selectVersion(e.version));
  $("messages").appendChild(btn);
  scrollChat();
}

function setStatus(text) {
  $("status-line").hidden = !text;
  if (text) $("status-text").textContent = text;
}

function setMode(mode) {
  state.mode = mode;
  localStorage.setItem("studio-mode", mode);
  $("mode-codegen").classList.toggle("active", mode === "codegen");
  $("mode-abc").classList.toggle("active", mode === "abc");
}
$("mode-codegen").addEventListener("click", () => setMode("codegen"));
$("mode-abc").addEventListener("click", () => setMode("abc"));
setMode(state.mode);

$("turn-model").addEventListener("change", () => {
  $("chat-model").textContent = $("turn-model").value;
});

$("composer").addEventListener("submit", (ev) => {
  ev.preventDefault();
  send();
});
$("input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    send();
  }
});

// True while the chat view is showing this session — the only time a stream
// may touch the chat DOM. Streams keep running (and logging server-side) after
// the user navigates away; reopening the session replays the log.
function viewingChat(sid) {
  return state.session && state.session.id === sid && !$("chat-view").hidden;
}

async function send() {
  const text = $("input").value.trim();
  if (!text || !state.session) return;
  const sid = state.session.id;
  if (state.streamingChats.has(sid)) return;
  $("input").value = "";
  state.streamingChats.add(sid);
  $("send").disabled = true;
  addUserBubble(text);
  setStatus("Thinking…");
  try {
    const res = await fetch(`/api/sessions/${sid}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode: state.mode, model: $("turn-model").value }),
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
      throw new Error(detail);
    }
    await readStream(res, sid);
  } catch (e) {
    if (viewingChat(sid)) addError(e.message);
  } finally {
    state.streamingChats.delete(sid);
    if (viewingChat(sid)) {
      $("send").disabled = false;
      setStatus(null);
      $("input").focus();
    }
  }
}

async function readStream(res, sid) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let bubble = null;   // current assistant div
  let raw = "";        // its accumulated markdown source
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      const line = part.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;  // ping comments
      const e = JSON.parse(line.slice(6));
      // Another session may be on screen now — drop the event; the log has it.
      if (!viewingChat(sid)) continue;
      if (e.type === "text_start") {
        bubble = addAssistantHtml("");
        raw = "";
        setStatus(null);
      } else if (e.type === "text") {
        // Recreate the bubble if a reopen rebuilt the DOM under us.
        if (!bubble || !bubble.isConnected) bubble = addAssistantHtml("");
        raw += e.delta;
        bubble.innerHTML = md(raw);
        scrollChat();
        setStatus(null);
      } else if (e.type === "status") {
        setStatus(e.text);
      } else if (e.type === "retry") {
        if (bubble) { bubble.remove(); bubble = null; raw = ""; }
      } else if (e.type === "piece") {
        registerPiece(e, true);
        addPieceCard(e);
        setStatus(null);
        bubble = null;
      } else if (e.type === "error") {
        addError(e.message);
        setStatus(null);
      } else if (e.type === "done") {
        setStatus(null);
      }
    }
  }
}

// ---------- pieces ----------

function fileUrl(version, name) {
  return `/api/sessions/${state.session.id}/pieces/v${version}/${name}`;
}

function registerPiece(e, select) {
  state.pieces.set(e.version, e);
  if (!$(`tab-v${e.version}`)) {  // a reopen mid-stream may have replayed it already
    const tab = document.createElement("button");
    tab.id = `tab-v${e.version}`;
    tab.textContent = `v${e.version}`;
    tab.title = e.title;
    tab.addEventListener("click", () => selectVersion(e.version));
    $("version-tabs").appendChild(tab);
  }
  if (select) selectVersion(e.version);
}

async function selectVersion(version) {
  const e = state.pieces.get(version);
  if (!e) return;
  state.currentVersion = version;
  $("piece-empty").hidden = true;
  $("piece-panel").hidden = false;
  for (const btn of $("version-tabs").children) {
    btn.classList.toggle("active", btn.id === `tab-v${version}`);
  }
  $("piece-title").textContent = `v${version} — ${e.title}`;
  const modeLabel = MODE_LABELS[e.mode] || e.mode;
  $("piece-note").textContent = [modeLabel, e.model, e.note].filter(Boolean).join(" · ");
  renderComments();
  const wrap = $("piece-features-wrap");
  wrap.innerHTML = "";
  wrap.appendChild(makeFeaturesDetails(state.session.id, version));

  const audio = $("piece-audio");
  if (e.files.includes("piece.mp3")) {
    audio.hidden = false;
    audio.src = fileUrl(version, "piece.mp3");
  } else {
    audio.hidden = true;
    audio.removeAttribute("src");
  }

  const score = $("score");
  score.innerHTML = '<p class="muted">Engraving…</p>';
  try {
    if (e.files.includes("piece.musicxml")) {
      const xml = await (await fetch(fileUrl(version, "piece.musicxml"))).text();
      await renderMusicXml(score, xml, version);
    } else if (e.files.includes("piece.abc")) {
      const abc = await (await fetch(fileUrl(version, "piece.abc"))).text();
      if (state.currentVersion !== version) return;
      score.innerHTML = "";
      ABCJS.renderAbc(score, abc, { responsive: "resize" });
    } else {
      score.innerHTML = '<p class="muted">No score for this version.</p>';
    }
  } catch (err) {
    score.innerHTML = `<p class="error">Could not engrave the score: ${esc(String(err))}</p>`;
  }
}

// ---------- system prompt editor (advanced) ----------

function setPromptEditorOpen(open) {
  $("prompt-editor").hidden = !open;
  $("messages").hidden = open;
  document.querySelector(".mode-bar").style.display = open ? "none" : "";
  $("composer").style.display = open ? "none" : "";
  if (open) $("status-line").hidden = true;
}

function markPromptState(custom) {
  $("prompt-btn").classList.toggle("custom-on", !!custom);
  $("prompt-state").textContent = custom
    ? "Custom prompt active for this session."
    : "Using the default prompt.";
}

$("prompt-btn").addEventListener("click", async () => {
  if (!state.session) return;
  const opening = $("prompt-editor").hidden;
  setPromptEditorOpen(opening);
  if (!opening) return;
  $("prompt-text").value = "Loading…";
  try {
    const r = await api(`/api/sessions/${state.session.id}/prompt?mode=${state.mode}`);
    $("prompt-text").value = r.custom || r.default;
    markPromptState(r.custom);
  } catch (e) {
    $("prompt-text").value = "";
    $("prompt-state").textContent = "Couldn't load the prompt: " + e.message;
  }
});

$("prompt-cancel").addEventListener("click", () => setPromptEditorOpen(false));

$("prompt-save").addEventListener("click", async () => {
  if (!state.session) return;
  try {
    const r = await api(`/api/sessions/${state.session.id}/prompt`, {
      method: "PUT", body: JSON.stringify({ text: $("prompt-text").value }),
    });
    markPromptState(r.custom);
    setPromptEditorOpen(false);
  } catch (e) {
    $("prompt-state").textContent = "Couldn't save: " + e.message;
  }
});

$("prompt-reset").addEventListener("click", async () => {
  if (!state.session) return;
  try {
    await api(`/api/sessions/${state.session.id}/prompt`, {
      method: "PUT", body: JSON.stringify({ text: null }),
    });
    const r = await api(`/api/sessions/${state.session.id}/prompt?mode=${state.mode}`);
    $("prompt-text").value = r.default;
    markPromptState(null);
  } catch (e) {
    $("prompt-state").textContent = "Couldn't reset: " + e.message;
  }
});

// ---------- measured features (same metrics as the batch analysis) ----------

const FEATURE_SKIP = new Set(["model", "prompt", "mode", "sample", "batch",
                              "title", "features_version"]);

function makeFeaturesDetails(sid, version) {
  const details = document.createElement("details");
  details.className = "features";
  details.innerHTML = "<summary>Measured features</summary>" +
    '<div class="features-body"></div>';
  details.addEventListener("toggle", async () => {
    if (!details.open || details.dataset.loaded) return;
    details.dataset.loaded = "1";
    const body = details.querySelector(".features-body");
    body.innerHTML = '<p class="muted">Measuring…</p>';
    try {
      const r = await api(`/api/sessions/${sid}/features/${version}`);
      if (!r.ok) { body.innerHTML = `<p class="muted">${esc(r.error)}</p>`; return; }
      const rows = Object.entries(r.features)
        .filter(([k]) => !FEATURE_SKIP.has(k))
        .map(([k, v]) => `<tr><td>${esc(k)}</td>` +
          `<td>${v === null || v === "" ? "—" : esc(String(v))}</td></tr>`);
      body.innerHTML = `<table class="features-table">${rows.join("")}</table>`;
    } catch (e) {
      body.innerHTML = `<p class="error">${esc(e.message)}</p>`;
    }
  });
  return details;
}

// ---------- comments (the composer's own notes; never sent to a model) ----------

function renderComments() {
  const list = state.comments.get(state.currentVersion) || [];
  $("piece-comments").innerHTML = list.map(
    (t) => `<p class="comment">${esc(t)}</p>`).join("");
}

$("comment-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = $("comment-input").value.trim();
  if (!text || !state.session || state.currentVersion == null) return;
  const sid = state.session.id;
  const version = state.currentVersion;
  $("comment-input").value = "";
  try {
    await api(`/api/sessions/${sid}/comments`, {
      method: "POST", body: JSON.stringify({ text, version }),
    });
    recordComment({ text, version });
    if (state.session && state.session.id === sid) renderComments();
  } catch (e) {
    alert("Couldn't save the note: " + e.message);
  }
});

async function saveCellComment(compId, version, text) {
  await api(`/api/sessions/${compId}/comments`, {
    method: "POST", body: JSON.stringify({ text, version }),
  });
}

// Verovio's wasm runtime initializes asynchronously; poll until the toolkit
// can be constructed (the onRuntimeInitialized hook races with deferred scripts).
let _toolkit = null;
function getToolkit() {
  if (_toolkit) return Promise.resolve(_toolkit);
  return new Promise((resolve, reject) => {
    const started = Date.now();
    const timer = setInterval(() => {
      try {
        _toolkit = new verovio.toolkit();
        clearInterval(timer);
        resolve(_toolkit);
      } catch (e) {
        if (Date.now() - started > 20000) {
          clearInterval(timer);
          reject(new Error("Verovio failed to load"));
        }
      }
    }, 200);
  });
}

async function renderMusicXml(container, xml, version) {
  const tk = await getToolkit();
  if (state.currentVersion !== version) return;
  const scale = 38;
  tk.setOptions({
    scale,
    pageWidth: Math.max(750, container.clientWidth - 20) * (100 / scale),
    adjustPageHeight: true,
    footer: "none",
  });
  tk.loadData(xml);
  let svgs = "";
  for (let p = 1; p <= tk.getPageCount(); p++) svgs += tk.renderToSVG(p);
  container.innerHTML = svgs;
}

// Engrave one piece into an arbitrary container (used by the comparison grid;
// no global race-guard — each call renders once). Verovio's toolkit is a single
// shared instance, so callers await this sequentially to avoid interleaving.
async function engraveInto(container, sessionId, version, files, scale) {
  const url = (name) => `/api/sessions/${sessionId}/pieces/v${version}/${name}`;
  container.innerHTML = '<p class="muted">Engraving…</p>';
  try {
    if (files.includes("piece.musicxml")) {
      const xml = await (await fetch(url("piece.musicxml"))).text();
      const tk = await getToolkit();
      tk.setOptions({ scale, footer: "none", adjustPageHeight: true,
        pageWidth: Math.max(520, container.clientWidth - 12) * (100 / scale) });
      tk.loadData(xml);
      let svgs = "";
      for (let p = 1; p <= tk.getPageCount(); p++) svgs += tk.renderToSVG(p);
      container.innerHTML = svgs;
    } else if (files.includes("piece.abc")) {
      const abc = await (await fetch(url("piece.abc"))).text();
      container.innerHTML = "";
      ABCJS.renderAbc(container, abc, { responsive: "resize", staffwidth: 480 });
    } else {
      container.innerHTML = '<p class="muted">No score.</p>';
    }
  } catch (err) {
    container.innerHTML = `<p class="error">Could not engrave: ${esc(String(err))}</p>`;
  }
}

// ---------- comparison ----------

const MAX_CMP_MODELS = 5;  // mirrors compare.MAX_MODELS server-side

function modesFromMethod(v) { return v === "both" ? ["codegen", "abc"] : [v]; }

$("cmp-back").addEventListener("click", showHome);

// One piece at a time: starting any player in the grid pauses the others.
// ('play' doesn't bubble, so listen in the capture phase.)
$("cmp-grid").addEventListener("play", (ev) => {
  for (const a of $("cmp-grid").querySelectorAll("audio")) {
    if (a !== ev.target) a.pause();
  }
}, true);

$("new-comparison-form").addEventListener("submit", (ev) => {
  ev.preventDefault();
  $("cmp-error").hidden = true;
  const prompt = $("cmp-prompt").value.trim();
  const models = [...document.querySelectorAll("#cmp-models input:checked")].map((c) => c.value);
  const method = document.querySelector("input[name=cmp-method]:checked").value;
  if (!prompt) return cmpError("Enter a prompt for the models to compose.");
  if (!models.length) return cmpError("Pick at least one model.");
  if (models.length > MAX_CMP_MODELS) {
    return cmpError(`You picked ${models.length} models — the limit is ` +
      `${MAX_CMP_MODELS} per comparison; more than that is hard to compare side by side.`);
  }
  startComparison(prompt, models, modesFromMethod(method), $("cmp-title").value.trim());
});

function cmpError(msg) {
  $("cmp-error").textContent = msg;
  $("cmp-error").hidden = false;
}

async function startComparison(prompt, models, modes, title) {
  let res;
  try {
    res = await fetch("/api/comparisons", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, models, modes, title }),
    });
  } catch (err) {
    return cmpError("Network error: " + err.message);
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
    return cmpError(detail);
  }
  $("cmp-prompt").value = "";
  $("cmp-title").value = "";
  try {
    await streamComparison(res);
  } catch (err) {
    // The server run is fire-and-forget, so a dropped stream loses nothing —
    // results keep landing in the event log and reopening shows them.
    if ($("compare-view").hidden) {
      cmpError("Connection lost: " + err.message);
    } else {
      $("cmp-view-status").textContent = "connection lost — models are still composing; reopen from History for results";
    }
  }
}

async function streamComparison(res, initialCompId = null) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let compId = initialCompId;  // this stream's session; fillCell ignores stale grids
  state.cmpStreaming = true;
  $("cmp-send").disabled = true;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split("\n\n");
      buf = parts.pop();
      for (const part of parts) {
        const line = part.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        const e = JSON.parse(line.slice(6));
        if (e.type === "created") {
          compId = e.meta.id;
          state.session = e.meta;
          state.cmpComments = new Map();
          $("cmp-view-title").textContent = e.meta.title;
          $("cmp-view-status").textContent = "composing…";
          $("cmp-rounds").innerHTML = "";
          show("compare-view");
        } else if (e.type === "start") {
          if (e.round) {
            addRoundEcho(e.prompt);
            resetCellsComposing(compId);
          } else {
            buildGrid(compId, e.cells);
            addRoundEcho(e.prompt);
          }
          $("cmp-view-status").textContent = "composing…";
        } else if (e.type === "cell") {
          await fillCell(compId, e);  // await: serialize engraving
        } else if (e.type === "done") {
          if ($("cmp-grid").dataset.session === compId) {
            $("cmp-view-status").textContent = e.n_ok === e.n_cells
              ? `${e.n_cells} result${e.n_cells === 1 ? "" : "s"}`
              : `${e.n_ok} of ${e.n_cells} rendered`;
          }
        } else if (e.type === "error") {
          $("cmp-view-status").textContent = "error";
          cmpError(e.message);
        }
      }
    }
  } finally {
    state.cmpStreaming = false;
    $("cmp-send").disabled = false;
  }
}

function addRoundEcho(text) {
  const p = document.createElement("p");
  p.className = "cmp-prompt-echo";
  p.textContent = text;
  $("cmp-rounds").appendChild(p);
}

function resetCellsComposing(compId) {
  if ($("cmp-grid").dataset.session !== compId) return;
  for (const cell of $("cmp-grid").children) {
    cell.querySelector(".cell-body").innerHTML = '<p class="muted">composing…</p>';
  }
}

function cellLabel(cell) {
  return `${esc(cell.model)} <span class="cell-method">${MODE_LABELS[cell.mode] || cell.mode}</span>`;
}

function buildGrid(compId, cells) {
  const grid = $("cmp-grid");
  grid.dataset.session = compId;  // late stream events check this before touching DOM
  grid.innerHTML = "";
  cells.forEach((cell, i) => {
    const div = document.createElement("div");
    div.className = "cmp-cell";
    div.id = `cell-${i}`;
    div.innerHTML =
      `<div class="cell-head"><span class="cell-label">${cellLabel(cell)}</span>` +
      `<span class="cell-pills"></span></div>` +
      `<div class="cell-body"><p class="muted">composing…</p></div>`;
    grid.appendChild(div);
  });
}

async function fillCell(compId, e) {
  // The stream outlives the view: if the user navigated to another comparison,
  // this grid's cells belong to someone else now — drop the event.
  if ($("cmp-grid").dataset.session !== compId) return;
  const cell = $(`cell-${e.index}`);
  if (!cell) return;
  if (!e.ok) {
    cell.querySelector(".cell-label").innerHTML = cellLabel(e);
    cell.querySelector(".cell-body").innerHTML =
      `<p class="error">Didn't render: ${esc(e.error || "unknown")}</p>`;
    return;
  }
  // Each successful round adds a version pill; the newest becomes active.
  const pills = cell.querySelector(".cell-pills");
  const pill = document.createElement("button");
  pill.className = "pill";
  pill.textContent = `v${pills.children.length + 1}`;
  pill.addEventListener("click", () => showCellResult(compId, e, cell, pill));
  pills.appendChild(pill);
  await showCellResult(compId, e, cell, pill);
}

async function showCellResult(compId, e, cell, pill) {
  if ($("cmp-grid").dataset.session !== compId) return;
  for (const p of cell.querySelectorAll(".cell-pills .pill")) {
    p.classList.toggle("active", p === pill);
  }
  cell.querySelector(".cell-label").innerHTML = cellLabel(e) +
    (e.title ? ` <span class="cell-title">${esc(e.title)}</span>` : "");
  const body = cell.querySelector(".cell-body");
  body.innerHTML = "";
  if (e.files.includes("piece.mp3")) {
    const audio = document.createElement("audio");
    audio.controls = true; audio.preload = "metadata";
    audio.src = `/api/sessions/${compId}/pieces/v${e.version}/piece.mp3`;
    body.appendChild(audio);
  }
  if (e.short_description) {
    const desc = document.createElement("p");
    desc.className = "cell-desc";
    desc.textContent = e.short_description;
    body.appendChild(desc);
  }
  const score = document.createElement("div");
  score.className = "cell-score";
  body.appendChild(score);

  const iterate = document.createElement("button");
  iterate.className = "cell-iterate";
  iterate.textContent = "Iterate in chat →";
  iterate.title = `Open a chat with ${e.model} continuing this cell's whole conversation`;
  iterate.addEventListener("click", () => forkCell(compId, e.index, iterate));
  body.appendChild(iterate);

  body.appendChild(makeFeaturesDetails(compId, e.version));

  const comments = document.createElement("div");
  comments.className = "cell-comments";
  const renderCellComments = () => {
    comments.innerHTML = (state.cmpComments.get(e.version) || []).map(
      (t) => `<p class="comment">${esc(t)}</p>`).join("");
  };
  renderCellComments();
  body.appendChild(comments);

  const note = document.createElement("form");
  note.className = "cell-note";
  note.innerHTML = `<input placeholder="Your thoughts — saved for the record, never sent to the model">` +
    `<button class="ghost" type="submit">Save</button>`;
  note.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const input = note.querySelector("input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    try {
      await saveCellComment(compId, e.version, text);
      if (!state.cmpComments.has(e.version)) state.cmpComments.set(e.version, []);
      state.cmpComments.get(e.version).push(text);
      renderCellComments();
    } catch (err) {
      alert("Couldn't save the note: " + err.message);
    }
  });
  body.appendChild(note);

  await engraveInto(score, compId, e.version, e.files, 30);
}

async function forkCell(compId, index, btn) {
  btn.disabled = true;
  btn.textContent = "Opening chat…";
  try {
    const chat = await api(`/api/sessions/${compId}/cells/${index}/fork`,
      { method: "POST" });
    openSession(chat.id);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = "Iterate in chat →";
    alert("Couldn't open a chat: " + e.message);
  }
}

$("cmp-composer").addEventListener("submit", (ev) => {
  ev.preventDefault();
  sendToAll();
});
$("cmp-input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    sendToAll();
  }
});

async function sendToAll() {
  const text = $("cmp-input").value.trim();
  if (!text || state.cmpStreaming || !state.session) return;
  $("cmp-input").value = "";
  try {
    const res = await fetch(`/api/sessions/${state.session.id}/compare-message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
      throw new Error(detail);
    }
    await streamComparison(res, state.session.id);
  } catch (e) {
    $("cmp-view-status").textContent =
      "connection lost — models are still composing; reopen from History for results";
    alert("Round failed: " + e.message);
  }
}

async function openComparison(id) {
  const { meta, events } = await api(`/api/sessions/${id}`);
  state.session = meta;
  $("cmp-view-title").textContent = meta.title;
  const prompts = events.filter((e) => e.type === "prompt");
  const cells = prompts.length ? prompts[0].cells : [];
  state.cmpComments = new Map();
  for (const e of events) {
    if (e.type !== "comment") continue;
    const v = e.version ?? 0;
    if (!state.cmpComments.has(v)) state.cmpComments.set(v, []);
    state.cmpComments.get(v).push(e.text);
  }
  $("cmp-rounds").innerHTML = "";
  buildGrid(id, cells);
  // Honest status: distinguish a finished grid from one whose run was
  // interrupted (or is still going) — those cells have no event yet.
  const cellEvents = events.filter((e) => e.type === "cell");
  const nOk = cellEvents.filter((e) => e.ok).length;
  const expected = prompts.length * cells.length;
  $("cmp-view-status").textContent =
    cellEvents.length < expected
      ? `${cellEvents.length} of ${expected} finished — reopen for updates`
      : (nOk === expected ? `${expected} result${expected === 1 ? "" : "s"}`
                          : `${nOk} of ${expected} rendered`);
  show("compare-view");
  // Chronological replay keeps each cell's pills in round order.
  for (const e of events) {
    if (e.type === "prompt") addRoundEcho(e.text);
    else if (e.type === "cell") await fillCell(id, e);
  }
}

init();
