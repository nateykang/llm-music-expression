/* Static listening page: the zero-server fallback of the studio's Listen tab.
   Pieces come from data.js (built by scripts/build_static_listen.py); notes
   live in localStorage per listener name until exported as a JSON file.
   Blinding is soft: the letter->model mapping ships base64-encoded in data.js
   and decodes only on an explicit per-window reveal. */

"use strict";

const $ = (id) => document.getElementById(id);
const DATA = window.LISTEN_DATA;
const STORE_KEY = "llm-music-listen-v1";

function loadStore() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY)) || { current: "", users: {} };
  } catch (e) {
    return { current: "", users: {} };
  }
}
function saveStore() { localStorage.setItem(STORE_KEY, JSON.stringify(store)); }

const store = loadStore();

function me() {
  if (!store.current) return null;
  if (!store.users[store.current]) {
    store.users[store.current] = { notes: [], revealed: {} };
  }
  return store.users[store.current];
}

function esc(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---------- identity ----------

function refreshUserBox() {
  const has = !!store.current;
  $("user-setup").hidden = has;
  $("user-chip").hidden = !has;
  $("user-change").hidden = !has;
  $("export-btn").hidden = !has;
  if (has) $("user-chip").textContent = store.current;
  $("gate").hidden = has;
  $("intro").hidden = !has;
  $("windows").hidden = !has;
  $("finish").hidden = !has;
  if (has) renderWindows();
}

$("user-save").addEventListener("click", () => {
  const name = $("user-input").value.trim().replace(/\s+/g, " ").slice(0, 40);
  if (!name) return;
  store.current = name;
  me();
  saveStore();
  refreshUserBox();
});
$("user-input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") { ev.preventDefault(); $("user-save").click(); }
});
$("user-change").addEventListener("click", () => {
  $("user-input").value = store.current;
  store.current = "";
  saveStore();
  refreshUserBox();
  $("user-input").focus();
});

// ---------- windows ----------

function noteHtml(n) {
  return `<p class="comment">${esc(n.text)}` +
    (n.revealed ? ' <span class="after-reveal">after reveal</span>' : "") + "</p>";
}

function notesFor(winId, piece) {
  const u = me();
  return u ? u.notes.filter((n) => n.w === winId && n.piece === piece) : [];
}

function groupsKey() {
  return JSON.parse(atob(DATA.key));
}

function renderWindows() {
  const u = me();
  const root = $("windows");
  root.innerHTML = "";
  DATA.windows.forEach((w) => {
    const revealed = !!u.revealed[w.id];
    const sec = document.createElement("section");
    sec.className = "card win";
    sec.id = w.id;
    const state = revealed ? "models shown" : "blind";
    sec.innerHTML = `<h2>${esc(w.title)} <span class="chip state">${state}</span></h2>`;
    if (revealed) {
      const legend = document.createElement("p");
      legend.className = "legend";
      legend.textContent = Object.entries(groupsKey())
        .map(([g, m]) => `${g} = ${m}`).join(" · ");
      sec.appendChild(legend);
    }
    w.pieces.forEach((p) => sec.appendChild(pieceEl(w, p, revealed)));

    const wn = document.createElement("div");
    wn.className = "win-notes";
    wn.innerHTML = '<div class="lbl">Comparing notes for this window</div>';
    const list = document.createElement("div");
    list.innerHTML = notesFor(w.id, null).map(noteHtml).join("");
    wn.appendChild(list);
    wn.appendChild(noteForm(w.id, null, list,
      "Overall thoughts across this window…"));
    sec.appendChild(wn);

    if (!revealed) {
      const foot = document.createElement("div");
      foot.className = "win-foot";
      const btn = document.createElement("button");
      btn.className = "ghost";
      btn.textContent = "Reveal models";
      btn.title = "Best saved for after your notes";
      btn.addEventListener("click", () => {
        if (!confirm("Reveal which model wrote each piece in this window? " +
                     "Notes you write afterwards are marked as post-reveal.")) return;
        me().revealed[w.id] = true;
        saveStore();
        renderWindows();
        document.getElementById(w.id).scrollIntoView();
      });
      foot.appendChild(btn);
      sec.appendChild(foot);
    }
    root.appendChild(sec);
  });
  refreshNoteCount();
}

function pieceEl(w, p, revealed) {
  const div = document.createElement("div");
  div.className = "piece";
  const who = revealed ? (groupsKey()[p.group] || p.group) : p.group;
  div.innerHTML =
    `<div class="piece-head"><span class="plabel">Piece ${p.n}</span>` +
    `<span class="chip">${esc(who)}</span>` +
    `<span class="mchip">${esc(p.mode)}</span>` +
    `<span class="mchip">${esc(p.prompt)}</span>` +
    (p.title ? `<span class="ptitle">${esc(p.title)}</span>` : "") + `</div>`;

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "metadata";
  audio.src = p.audio;
  div.appendChild(audio);

  const det = document.createElement("details");
  det.innerHTML = "<summary>Score</summary>";
  const scoreDiv = document.createElement("div");
  scoreDiv.className = "score";
  det.appendChild(scoreDiv);
  det.addEventListener("toggle", () => {  // engrave lazily, once
    if (!det.open || det.dataset.engraved) return;
    det.dataset.engraved = "1";
    engrave(scoreDiv, p);
  });
  div.appendChild(det);

  const list = document.createElement("div");
  list.innerHTML = notesFor(w.id, p.n - 1).map(noteHtml).join("");
  div.appendChild(list);
  div.appendChild(noteForm(w.id, p.n - 1, list, "Your thoughts on this piece…"));
  return div;
}

function noteForm(winId, piece, listEl, placeholder) {
  const form = document.createElement("form");
  form.className = "note-form";
  form.innerHTML =
    `<textarea rows="2" placeholder="${placeholder}"></textarea>` +
    `<button type="submit" class="ghost">Save note</button>`;
  form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    const input = form.querySelector("textarea");
    const text = input.value.trim();
    if (!text) return;
    const u = me();
    if (!u) return;
    const note = { w: winId, piece, text,
                   revealed: !!u.revealed[winId], ts: new Date().toISOString() };
    u.notes.push(note);
    saveStore();
    input.value = "";
    listEl.insertAdjacentHTML("beforeend", noteHtml(note));
    refreshNoteCount();
  });
  form.querySelector("textarea").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); form.requestSubmit(); }
  });
  return form;
}

function refreshNoteCount() {
  const u = me();
  const n = u ? u.notes.length : 0;
  $("note-count").textContent = `${n} note${n === 1 ? "" : "s"}`;
}

// One piece at a time: starting any player pauses the others.
document.addEventListener("play", (ev) => {
  for (const a of document.querySelectorAll("audio")) {
    if (a !== ev.target) a.pause();
  }
}, true);

// ---------- engraving ----------

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

async function engrave(container, p) {
  container.innerHTML = '<p class="muted">Engraving…</p>';
  try {
    if (p.score) {
      const xml = await (await fetch(p.score)).text();
      const tk = await getToolkit();
      const scale = 34;
      tk.setOptions({ scale, footer: "none", adjustPageHeight: true,
        pageWidth: Math.max(700, container.clientWidth - 16) * (100 / scale) });
      tk.loadData(xml);
      let svgs = "";
      for (let pg = 1; pg <= tk.getPageCount(); pg++) svgs += tk.renderToSVG(pg);
      container.innerHTML = svgs;
    } else if (p.abc) {
      container.innerHTML = "";
      ABCJS.renderAbc(container, p.abc, { responsive: "resize" });
    } else {
      container.innerHTML = '<p class="muted">No score for this piece.</p>';
    }
  } catch (err) {
    container.innerHTML = `<p class="error">Could not engrave: ${esc(String(err))}</p>`;
  }
}

// ---------- export ----------

function exportNotes() {
  const u = me();
  if (!u) return;
  const payload = {
    name: store.current,
    exported_at: new Date().toISOString(),
    suite: DATA.suite,
    revealed: u.revealed,
    notes: u.notes,
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const stamp = new Date().toISOString().slice(0, 10);
  a.download = `listening-notes-${store.current.replace(/\s+/g, "_")}-${stamp}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}
$("export-btn").addEventListener("click", exportNotes);
$("export-btn-2").addEventListener("click", exportNotes);

refreshUserBox();
