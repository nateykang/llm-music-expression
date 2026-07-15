/* Composer Studio frontend: login -> session list -> chat with streaming turns.
   Pieces stream in as SSE events; MusicXML engraves via Verovio, ABC via abcjs. */

"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  models: [],
  defaultModel: "",
  session: null,          // meta of the open session
  pieces: new Map(),      // version (number) -> piece event
  currentVersion: null,
  streaming: false,
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
  for (const v of ["login-view", "home-view", "chat-view"]) $(v).hidden = v !== view;
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
    const btn = document.createElement("button");
    btn.className = "session-item";
    btn.innerHTML =
      `<div class="title">${esc(s.title)}</div>` +
      `<div class="meta">${esc(s.model)} &middot; ` +
      `${new Date(s.last_active * 1000).toLocaleString()} &middot; ` +
      `${s.n_versions} version${s.n_versions === 1 ? "" : "s"}</div>`;
    btn.addEventListener("click", () => openSession(s.id));
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
  }
  const last = Math.max(0, ...state.pieces.keys());
  if (last) selectVersion(last);
  show("chat-view");
  scrollChat();
  $("input").focus();
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

async function send() {
  const text = $("input").value.trim();
  if (!text || state.streaming || !state.session) return;
  $("input").value = "";
  state.streaming = true;
  $("send").disabled = true;
  addUserBubble(text);
  setStatus("Thinking…");
  try {
    const res = await fetch(`/api/sessions/${state.session.id}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, mode: state.mode, model: $("turn-model").value }),
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (e) { /* ignore */ }
      throw new Error(detail);
    }
    await readStream(res);
  } catch (e) {
    addError(e.message);
  } finally {
    state.streaming = false;
    $("send").disabled = false;
    setStatus(null);
    $("input").focus();
  }
}

async function readStream(res) {
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
      if (e.type === "text_start") {
        bubble = addAssistantHtml("");
        raw = "";
        setStatus(null);
      } else if (e.type === "text") {
        if (!bubble) { bubble = addAssistantHtml(""); raw = ""; }
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
  const tab = document.createElement("button");
  tab.id = `tab-v${e.version}`;
  tab.textContent = `v${e.version}`;
  tab.title = e.title;
  tab.addEventListener("click", () => selectVersion(e.version));
  $("version-tabs").appendChild(tab);
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

init();
