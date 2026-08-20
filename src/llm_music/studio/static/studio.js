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
  cmpLatest: new Map(),   // cell index -> latest ok cell event (dashboard columns)
  currentVersion: null,
  review: null,           // open listening session: {id, revealed, notes, overall}
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
  // Leaving a view silences it — music shouldn't follow you to the homepage.
  for (const a of document.querySelectorAll("audio")) a.pause();
  for (const v of ["login-view", "home-view", "chat-view", "compare-view", "review-view"])
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

// Pickers show models grouped by family, alphabetical within each.
const FAMILY_RANK = { Claude: 0, GPT: 1, Gemini: 2, Grok: 3, Other: 4 };
function modelFamily(m) {
  if (/^(opus|sonnet|haiku|fable)/.test(m)) return "Claude";
  if (/^gpt/.test(m)) return "GPT";
  if (/^gemini/.test(m)) return "Gemini";
  if (/^grok/.test(m)) return "Grok";
  return "Other";
}
function sortModels(models) {
  return [...models].sort((a, b) =>
    (FAMILY_RANK[modelFamily(a)] - FAMILY_RANK[modelFamily(b)]) || a.localeCompare(b));
}

// Family-grouped checkbox grid, shared by the comparison and listening pickers.
function buildModelCheckgrid(grid, models, labelFn) {
  grid.innerHTML = "";
  let fam = null;
  for (const m of models) {
    if (modelFamily(m) !== fam) {
      fam = modelFamily(m);
      const h = document.createElement("div");
      h.className = "fam";
      h.textContent = fam;
      grid.appendChild(h);
    }
    const lbl = document.createElement("label");
    lbl.className = "chk";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = m;
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(" " + (labelFn ? labelFn(m) : m)));
    grid.appendChild(lbl);
  }
}

function afterLogin(me) {
  state.models = sortModels(me.models);
  state.defaultModel = me.default_model;
  const sel = $("new-model");
  sel.innerHTML = "";
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    if (m === me.default_model) opt.selected = true;
    sel.appendChild(opt);
  }
  buildModelCheckgrid($("cmp-models"), state.models);
  loadReviewBatches();  // fills the listening-session picker in the background
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
    const kind = s.kind || "chat";
    const btn = document.createElement("button");
    btn.className = "session-item";
    const tag = kind === "comparison" ? '<span class="kind">comparison</span>'
      : kind === "review" ? '<span class="kind">listening</span>' : "";
    const count = kind === "comparison"
      ? `${s.n_versions} result${s.n_versions === 1 ? "" : "s"}`
      : kind === "review"
      ? `${s.n_pieces ?? "?"} pieces · ${s.n_notes ?? 0} note${(s.n_notes ?? 0) === 1 ? "" : "s"}`
      : `${esc(s.model)} · ${s.n_versions} version${s.n_versions === 1 ? "" : "s"}`;
    btn.innerHTML =
      `<div class="title">${tag}${esc(s.title)}</div>` +
      `<div class="meta">${new Date(s.last_active * 1000).toLocaleString()} · ${count}</div>`;
    btn.addEventListener("click", () =>
      kind === "comparison" ? openComparison(s.id)
      : kind === "review" ? openReview(s.id) : openSession(s.id));
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
  if (![...sel.options].some((o) => o.value === meta.model)) {
    // Session on a model hidden from the picker (e.g. kimi-k3): keep it selectable here.
    const opt = document.createElement("option");
    opt.value = meta.model;
    opt.textContent = meta.model;
    sel.appendChild(opt);
  }
  sel.value = meta.model;
  $("messages").innerHTML = "";
  $("version-tabs").innerHTML = "";
  $("piece-panel").hidden = true;
  $("piece-empty").hidden = false;
  $("chat-dash").open = false;
  $("chat-dash-body").innerHTML = "";
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
  if ($("chat-dash").open) buildChatDashboard();  // keep an open dashboard current
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

// ---------- features dashboards (same metrics as the batch analysis) ----------
// One table per view — chat: feature rows × version columns; comparison:
// feature rows × model columns (each cell's latest render) — so pieces are
// compared side by side instead of one dropdown per piece.

const FEATURE_SKIP = new Set(["model", "prompt", "mode", "sample", "batch",
                              "title", "features_version"]);

async function fetchFeatures(sid, version) {
  try {
    const r = await api(`/api/sessions/${sid}/features/${version}`);
    return r.ok ? r.features : null;
  } catch (e) {
    return null;
  }
}

function dashTable(cols) {
  // cols: [{label (pre-escaped html), feats (dict|null)}]
  const keys = [];
  for (const c of cols) {
    if (!c.feats) continue;
    for (const k of Object.keys(c.feats)) {
      if (!FEATURE_SKIP.has(k) && !keys.includes(k)) keys.push(k);
    }
  }
  if (!keys.length) return '<p class="muted">Nothing analyzable yet.</p>';
  const head = "<tr><th></th>" + cols.map((c) => `<th>${c.label}</th>`).join("") + "</tr>";
  const rows = keys.map((k) => `<tr><td>${esc(k)}</td>` + cols.map((c) => {
    const v = c.feats ? c.feats[k] : undefined;
    return `<td>${v === null || v === undefined || v === "" ? "—" : esc(String(v))}</td>`;
  }).join("") + "</tr>").join("");
  return `<div class="dash-scroll"><table class="features-table dash-table">${head}${rows}</table></div>`;
}

async function buildChatDashboard() {
  const body = $("chat-dash-body");
  const versions = [...state.pieces.keys()].sort((a, b) => a - b);
  if (!versions.length) { body.innerHTML = '<p class="muted">Nothing rendered yet.</p>'; return; }
  body.innerHTML = '<p class="muted">Measuring…</p>';
  const sid = state.session.id;
  const cols = await Promise.all(versions.map(async (v) => ({
    label: `v${v}`, feats: await fetchFeatures(sid, v),
  })));
  body.innerHTML = dashTable(cols);
}

async function buildCmpDashboard() {
  const body = $("cmp-dash-body");
  const sid = $("cmp-grid").dataset.session;
  const entries = [...state.cmpLatest.entries()].sort((a, b) => a[0] - b[0]);
  if (!sid || !entries.length) { body.innerHTML = '<p class="muted">Nothing rendered yet.</p>'; return; }
  body.innerHTML = '<p class="muted">Measuring…</p>';
  const cols = await Promise.all(entries.map(async ([, e]) => ({
    label: `${esc(e.model)}<br><span class="cell-method">${MODE_LABELS[e.mode] || esc(e.mode)}</span>`,
    feats: await fetchFeatures(sid, e.version),
  })));
  body.innerHTML = dashTable(cols);
}

$("chat-dash").addEventListener("toggle", () => { if ($("chat-dash").open) buildChatDashboard(); });
$("cmp-dash").addEventListener("toggle", () => { if ($("cmp-dash").open) buildCmpDashboard(); });

// ---------- comments (the composer's own notes; never sent to a model) ----------

function renderComments() {
  const list = state.comments.get(state.currentVersion) || [];
  $("piece-comments").innerHTML = list.map(
    (t) => `<p class="comment">${esc(t)}</p>`).join("");
}

$("comment-input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    $("comment-form").requestSubmit();
  }
});

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

// Engrave into an arbitrary container from a MusicXML or ABC URL (used by the
// comparison grid and the listening view; no global race-guard — each call
// renders once). Verovio's toolkit is a single shared instance, so callers
// await this sequentially to avoid interleaving.
async function engraveSources(container, xmlUrl, abcUrl, scale) {
  container.innerHTML = '<p class="muted">Engraving…</p>';
  try {
    if (xmlUrl) {
      const xml = await (await fetch(xmlUrl)).text();
      const tk = await getToolkit();
      tk.setOptions({ scale, footer: "none", adjustPageHeight: true,
        pageWidth: Math.max(520, container.clientWidth - 12) * (100 / scale) });
      tk.loadData(xml);
      let svgs = "";
      for (let p = 1; p <= tk.getPageCount(); p++) svgs += tk.renderToSVG(p);
      container.innerHTML = svgs;
    } else if (abcUrl) {
      const abc = await (await fetch(abcUrl)).text();
      container.innerHTML = "";
      ABCJS.renderAbc(container, abc, { responsive: "resize", staffwidth: 480 });
    } else {
      container.innerHTML = '<p class="muted">No score.</p>';
    }
  } catch (err) {
    container.innerHTML = `<p class="error">Could not engrave: ${esc(String(err))}</p>`;
  }
}

async function engraveInto(container, sessionId, version, files, scale) {
  const url = (name) => `/api/sessions/${sessionId}/pieces/v${version}/${name}`;
  await engraveSources(container,
    files.includes("piece.musicxml") ? url("piece.musicxml") : null,
    files.includes("piece.abc") ? url("piece.abc") : null, scale);
}

// ---------- listening sessions (blind review of batch pieces) ----------
// No model is called anywhere here: pieces are already-generated batch output,
// and the composer's notes are research data. While blind, the server sends
// only "Model A/B/C" group labels; identities arrive after a logged reveal.

let reviewBatches = [];

function fmtBatchTs(ts) {
  const m = /^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})/.exec(ts || "");
  return m ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}` : ts;
}

async function loadReviewBatches() {
  const list = $("rev-batches");
  try {
    reviewBatches = (await api("/api/review/batches")).batches;
  } catch (e) {
    reviewBatches = [];
  }
  list.innerHTML = "";
  if (!reviewBatches.length) {
    list.innerHTML = '<p class="muted">No batches found.</p>';
    fillReviewModels();
    return;
  }
  reviewBatches.forEach((b, i) => {
    const n = Object.keys(b.model_counts).length;
    const lbl = document.createElement("label");
    lbl.className = "chk";
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = b.name;
    cb.checked = i === 0;  // newest pre-checked
    cb.addEventListener("change", fillReviewModels);
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(
      ` ${fmtBatchTs(b.timestamp)} — ${n} model${n === 1 ? "" : "s"}, ` +
      `${b.prompts.length} prompt${b.prompts.length === 1 ? "" : "s"}` +
      (b.modes.length ? ` (${b.modes.map((m) => MODE_LABELS[m] || m).join("+")})` : "")));
    list.appendChild(lbl);
  });
  fillReviewModels();
}

function checkedReviewBatches() {
  return [...document.querySelectorAll("#rev-batches input:checked")].map((c) => c.value);
}

function fillReviewModels() {
  const chosen = new Set(checkedReviewBatches());
  const counts = {};  // model -> total ok pieces across the chosen batches
  for (const b of reviewBatches) {
    if (!chosen.has(b.name)) continue;
    for (const [m, n] of Object.entries(b.model_counts)) {
      counts[m] = (counts[m] || 0) + n;
    }
  }
  buildModelCheckgrid($("rev-models"), sortModels(Object.keys(counts)),
    (m) => `${m} (${counts[m]})`);
}

function revError(msg) {
  $("rev-error").textContent = msg;
  $("rev-error").hidden = false;
}

$("new-review-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  $("rev-error").hidden = true;
  const batches = checkedReviewBatches();
  const models = [...document.querySelectorAll("#rev-models input:checked")].map((c) => c.value);
  if (!batches.length) return revError("Pick at least one batch.");
  if (!models.length) return revError("Pick at least one model.");
  const body = {
    batches, models,
    per_cell: parseInt($("rev-per-cell").value, 10) || 1,
    blind: $("rev-blind").checked,
    // A fresh seed per session so the slice varies; it's logged server-side,
    // so any queue stays reproducible.
    seed: Math.floor(Math.random() * 1e9),
    title: $("rev-title").value.trim(),
  };
  try {
    const res = await api("/api/reviews", { method: "POST", body: JSON.stringify(body) });
    $("rev-title").value = "";
    openReview(res.meta.id);
  } catch (e) {
    revError(e.message);
  }
});

$("rev-back").addEventListener("click", showHome);

// One piece at a time, same as the comparison grid.
$("rev-pieces").addEventListener("play", (ev) => {
  for (const a of $("rev-pieces").querySelectorAll("audio")) {
    if (a !== ev.target) a.pause();
  }
}, true);

async function openReview(id) {
  const r = await api(`/api/reviews/${id}`);
  state.review = { id, revealed: r.revealed, notes: new Map(), overall: [] };
  for (const n of r.notes) {
    if (n.piece == null) state.review.overall.push(n);
    else {
      if (!state.review.notes.has(n.piece)) state.review.notes.set(n.piece, []);
      state.review.notes.get(n.piece).push(n);
    }
  }
  $("rev-view-title").textContent = r.meta.title;
  $("rev-view-status").textContent =
    `${r.pieces.length} piece${r.pieces.length === 1 ? "" : "s"} · ` +
    (r.revealed ? "models shown" : "blind");
  $("rev-reveal").hidden = r.revealed;
  $("rev-hint").textContent = r.revealed
    ? Object.entries(r.groups || {}).map(([g, m]) => `${g} = ${m}`).join(" · ")
    : "Model names are hidden — pieces sharing a letter are by the same model. " +
      "Reveal when you're done listening.";
  const list = $("rev-pieces");
  list.innerHTML = "";
  for (const p of r.pieces) list.appendChild(reviewPieceCard(p));
  renderOverallNotes();
  show("review-view");
  window.scrollTo(0, 0);
}

function noteHtml(n) {
  return `<p class="comment">${esc(n.text)}` +
    (n.revealed ? ' <span class="after-reveal">after reveal</span>' : "") + "</p>";
}

function reviewPieceCard(p) {
  const sid = state.review.id;
  const card = document.createElement("div");
  card.className = "card rev-piece";
  const who = state.review.revealed && p.model ? p.model : p.group;
  const head = document.createElement("div");
  head.className = "rev-piece-head";
  head.innerHTML =
    `<span class="plabel">Piece ${p.idx + 1}</span>` +
    `<span class="chip">${esc(who)}</span>` +
    `<span class="cell-method">${MODE_LABELS[p.mode] || esc(p.mode)}</span>` +
    (p.prompt_label ? `<span class="cell-method">${esc(p.prompt_label)}</span>` : "") +
    (p.title ? `<span class="ptitle">${esc(p.title)}</span>` : "");
  card.appendChild(head);

  if (p.has_audio) {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = `/api/reviews/${sid}/pieces/${p.idx}/audio`;
    card.appendChild(audio);
  }

  const det = document.createElement("details");
  det.innerHTML = "<summary>Score</summary>";
  const scoreDiv = document.createElement("div");
  scoreDiv.className = "rev-score";
  det.appendChild(scoreDiv);
  det.addEventListener("toggle", () => {  // engrave lazily, once
    if (!det.open || det.dataset.engraved) return;
    det.dataset.engraved = "1";
    engraveSources(scoreDiv,
      p.has_score ? `/api/reviews/${sid}/pieces/${p.idx}/score` : null,
      p.has_abc ? `/api/reviews/${sid}/pieces/${p.idx}/abc` : null, 34);
  });
  card.appendChild(det);

  const comments = document.createElement("div");
  comments.className = "comments";
  const renderNotes = () => {
    comments.innerHTML = (state.review.notes.get(p.idx) || []).map(noteHtml).join("");
  };
  renderNotes();
  card.appendChild(comments);

  const form = document.createElement("form");
  form.className = "comment-form";
  form.innerHTML =
    `<textarea rows="2" placeholder="Your thoughts on this piece…"></textarea>` +
    `<button type="submit" class="ghost">Save note</button>`;
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const input = form.querySelector("textarea");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    try {
      await api(`/api/reviews/${sid}/notes`, {
        method: "POST", body: JSON.stringify({ text, piece: p.idx }),
      });
      if (!state.review.notes.has(p.idx)) state.review.notes.set(p.idx, []);
      state.review.notes.get(p.idx).push({ text, revealed: state.review.revealed });
      renderNotes();
    } catch (e) {
      alert("Couldn't save the note: " + e.message);
    }
  });
  form.querySelector("textarea").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); form.requestSubmit(); }
  });
  card.appendChild(form);
  return card;
}

function renderOverallNotes() {
  $("rev-overall-notes").innerHTML = state.review.overall.map(noteHtml).join("");
}

$("rev-overall-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (!state.review) return;
  const text = $("rev-overall-input").value.trim();
  if (!text) return;
  $("rev-overall-input").value = "";
  try {
    await api(`/api/reviews/${state.review.id}/notes`, {
      method: "POST", body: JSON.stringify({ text, piece: null }),
    });
    state.review.overall.push({ text, revealed: state.review.revealed });
    renderOverallNotes();
  } catch (e) {
    alert("Couldn't save the note: " + e.message);
  }
});
$("rev-overall-input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    $("rev-overall-form").requestSubmit();
  }
});

$("rev-reveal").addEventListener("click", async () => {
  if (!state.review) return;
  if (!confirm("Reveal which model wrote each piece? The reveal is logged, and " +
               "notes written afterwards are marked as post-reveal.")) return;
  try {
    await api(`/api/reviews/${state.review.id}/reveal`, { method: "POST" });
    openReview(state.review.id);  // refetch: names, groups legend, status chip
  } catch (e) {
    alert("Couldn't reveal: " + e.message);
  }
});

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
  state.cmpLatest = new Map();
  $("cmp-dash").open = false;
  $("cmp-dash-body").innerHTML = "";
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
  state.cmpLatest.set(e.index, e);
  if ($("cmp-dash").open) buildCmpDashboard();  // keep an open dashboard current
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
  note.innerHTML = `<textarea rows="2" placeholder="Your thoughts — saved for the record, never sent to the model"></textarea>` +
    `<button class="ghost" type="submit">Save</button>`;
  note.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const input = note.querySelector("textarea");
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
  note.querySelector("textarea").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); note.requestSubmit(); }
  });
  body.appendChild(note);

  const iterate = document.createElement("button");
  iterate.className = "cell-iterate";
  iterate.textContent = "Iterate in chat →";
  iterate.title = `Open a chat with ${e.model} continuing this cell's whole conversation`;
  iterate.addEventListener("click", () => forkCell(compId, e.index, iterate));
  body.appendChild(iterate);

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
