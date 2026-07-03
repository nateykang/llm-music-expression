"""Shared page framework for the generated dashboards (results / judge / audio).

Everything the three report generators have in common lives here: the site
palette, the HTML document skeleton (head, nav tabs, wrap, shared CSS/JS), the
sortable-table + tooltip + mode-toggle machinery, and the small formatting
helpers. A report module supplies only its sections and calls ``page()``.

The generated pages link docs/style.css and use its CSS variables (--bg, --fg,
--muted, --accent, --border) so all four site tabs share one palette. The hex
constants below mirror those variables for the one place CSS can't reach:
matplotlib charts.
"""

from __future__ import annotations

import html

# Site palette — keep in sync with the CSS variables in docs/style.css.
BG = "#faf9f7"
INK = "#1d1b18"
MUTED = "#6b665d"
ACCENT = "#7a5c3e"
CARD = "#ffffff"
BORDER = "#e4e0d8"

# Display shortening for model names in dense tables.
SHORT = {"gpt-5.5": "gpt-5.5", "gemini-2.5-pro": "gemini", "opus-4.8": "opus",
         "opus-4.8-thinking": "opus-think", "sonnet-4.6": "sonnet",
         "sonnet-4.6-thinking": "sonnet-think", "deepseek-v4-pro": "deepseek",
         "gpt-4.1": "gpt-4.1", "grok-4.3": "grok", "qwen3-max": "qwen",
         "llama-4-maverick": "llama"}

# Generation-mode toggle (judge/audio pages): ABC groups abc + smt-abc.
MODE_TOGGLE = [("abc", "ABC"), ("code", "code-gen"), ("all", "both")]
# Representation toggle (results page): ABC (declared K:) vs code-gen (detected).
REP_TOGGLE = [("text", "ABC"), ("code", "code-gen"), ("all", "both")]


def fnum(x) -> float | None:
    """Parse to float, or None (NaN -> None so it drops from averages)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def mode_filter(items, mode):
    """Filter raw pieces / feature rows by generation mode. ABC groups abc + smt-abc
    (both notation-based); code = code-gen; all = everything."""
    if mode == "all":
        return items
    if mode == "abc":
        return [x for x in items if x.get("mode") in ("abc", "smt-abc")]
    return [x for x in items if x.get("mode") == "codegen"]


def paned(fn, options=MODE_TOGGLE, default="all"):
    """Render content once per toggle option into panes; only `default` starts
    visible, the shared toggle JS switches the others in."""
    return "".join(
        f"<div class='mode-pane' data-mode='{key}'{'' if key == default else ' hidden'}>{fn(key)}</div>"
        for key, _ in options)


def toggle(options=MODE_TOGGLE, default="all", label="Generation", note=""):
    """The sticky pane-switcher matching ``paned()``. One per page."""
    btns = "".join(
        f"<button data-mode='{key}'{' aria-pressed=\"true\"' if key == default else ''}>{lbl}</button>"
        for key, lbl in options)
    note_html = f"<span class='note'>{note}</span>" if note else ""
    return (f"<div class='mode-toggle' data-default='{default}'>"
            f"<span class='lbl'>{label}</span>{btns}{note_html}</div>")


def group_by_model(rows) -> dict[str, list]:
    g: dict[str, list] = {}
    for r in rows:
        g.setdefault(r["model"], []).append(r)
    return g


# ---------- table building ----------

def tip(label, tip_text):
    """A <th>, with a hover/focus tooltip when tip_text is given."""
    if not tip_text:
        return f"<th>{html.escape(label)}</th>"
    return (f"<th><span class='tip' tabindex='0' data-tip=\"{html.escape(tip_text)}\">"
            f"{html.escape(label).replace(' ', '&nbsp;')}</span></th>")


def cell(v, kind):
    """Format a value for a table cell: text | int | pct | f0 | f1 | f2."""
    if v is None:
        return "—"
    if kind == "text":
        return html.escape(str(v))
    if kind == "int":
        return str(int(v))
    if kind == "pct":
        return f"{v * 100:.0f}%"
    if kind == "f0":
        return f"{v:.0f}"
    if kind == "f1":
        return f"{v:.1f}"
    return f"{v:.2f}"


def table(cols, rows, left=1):
    """Sortable table. cols = [(label, tip_or_None)]; each row is a list of
    cell-HTML strings, or a (cells, row_class) tuple (rows classed 'ref' stay
    pinned on top when sorting). The first `left` cells are left-aligned."""
    head = "<thead><tr>" + "".join(tip(l, t) for l, t in cols) + "</tr></thead>"
    body = "<tbody>"
    for row in rows:
        cells, cls = row if isinstance(row, tuple) else (row, "")
        body += (f"<tr class='{cls}'>" if cls else "<tr>") + "".join(
            f"<td class='{'m' if i < left else ''}'>{c}</td>" for i, c in enumerate(cells)) + "</tr>"
    body += "</tbody>"
    return f"<div class='tscroll'><table class='sortable'>{head}{body}</table></div>"


def heat(v, scale=0.55):
    """A heatmap <td>: green for positive, red for negative, opacity ∝ |v|."""
    if v is None:
        return "<td>—</td>"
    a = min(0.5, abs(v) / scale * 0.5)
    rgb = "46,160,67" if v >= 0 else "207,90,80"
    return f"<td style='background:rgba({rgb},{a:.2f})'>{v:+.2f}</td>"


# ---------- page skeleton ----------

TABS = [("index.html", "Browse outputs"), ("results.html", "Results &amp; analysis"),
        ("judge.html", "LLM judge"), ("audio.html", "Audio emotion")]

SHARED_CSS = """
  .wrap { max-width: 980px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
  .sub { color: var(--muted); font-weight: 400; font-size: .8em; }
  .scope { color: var(--muted); font-size: .9rem; margin: .25rem 0 1.25rem; }
  .tscroll { overflow-x: auto; }
  table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: .9rem; }
  th, td { text-align: right; padding: .35rem .55rem; border-bottom: 1px solid var(--border); }
  th { color: var(--muted); font-weight: 600; position: relative; }
  .tip { border-bottom: 1px dotted var(--muted); cursor: help; outline: none; }
  #tipbox { position: fixed; z-index: 100; width: 240px; white-space: normal; text-align: left;
    font-weight: 400; font-size: .76rem; line-height: 1.45; color: var(--bg); background: var(--fg);
    padding: .55rem .65rem; border-radius: 7px; box-shadow: 0 6px 20px rgba(0,0,0,.2);
    pointer-events: none; display: none; }
  td.m, th:first-child { text-align: left; font-weight: 600; }
  td.sub { color: var(--muted); font-size: .8rem; }
  tr.ref td { font-style: italic; color: var(--accent); background: #f3ede4; border-bottom: 2px solid #d8c9b5; }
  table.heat td { text-align: center; }
  h2 { margin-top: 2.4rem; }
  figure { margin: 1.5rem 0; }
  figcaption { color: var(--muted); font-size: .85rem; margin-top: .4rem; }
  .callout { background: #f3ede4; border-left: 3px solid var(--accent); padding: .7rem .9rem;
    border-radius: 0 7px 7px 0; font-size: .9rem; margin: .8rem 0 0; }
  .mode-toggle { position: sticky; top: 0; z-index: 10; background: var(--bg); display: flex;
    gap: 8px; align-items: center; padding: .6rem 0; margin-bottom: .5rem;
    border-bottom: 1px solid var(--border); }
  .mode-toggle .lbl { font-weight: 600; color: var(--fg); font-size: .9rem; }
  .mode-toggle .note { font-size: .8rem; color: var(--muted); }
  .mode-toggle button { font: inherit; font-size: .85rem; padding: 4px 13px; border-radius: 7px;
    border: 1px solid #cbb99a; background: #fff; color: var(--fg); cursor: pointer; }
  .mode-toggle button[aria-pressed=true] { background: var(--accent); color: var(--bg); border-color: var(--accent); }
  .mode-pane[hidden] { display: none; }
  table.sortable th { cursor: pointer; user-select: none; white-space: nowrap; }
  table.sortable th[data-dir=asc]::after { content: ' ▲'; font-size: .6em; opacity: .6; }
  table.sortable th[data-dir=desc]::after { content: ' ▼'; font-size: .6em; opacity: .6; }
"""

# Pane toggle (calls the optional window.__onModeChange hook, e.g. the key widget),
# column-sort with 'ref' rows pinned on top and empty cells sinking last, and the
# floating #tipbox for header tooltips.
SHARED_JS = r"""
(function(){
  var tog = document.querySelector('.mode-toggle');
  function setMode(m){
    document.querySelectorAll('.mode-pane').forEach(function(p){ p.hidden = p.dataset.mode !== m; });
    if (tog) tog.querySelectorAll('button').forEach(function(b){ b.setAttribute('aria-pressed', String(b.dataset.mode === m)); });
    if (window.__onModeChange) window.__onModeChange(m);
  }
  if (tog){
    tog.querySelectorAll('button').forEach(function(b){ b.addEventListener('click', function(){ setMode(b.dataset.mode); }); });
    setMode(tog.dataset.default || 'all');
  }

  function makeSortable(table){
    var head = table.tHead, body = table.tBodies[0];
    if (!head || !body) return;
    var cells = head.rows[0].cells;
    Array.prototype.forEach.call(cells, function(th, idx){
      th.addEventListener('click', function(){
        var asc = th.dataset.dir !== 'asc';
        Array.prototype.forEach.call(cells, function(h){ h.removeAttribute('data-dir'); });
        th.dataset.dir = asc ? 'asc' : 'desc';
        var num = function(c){ var n = parseFloat(c.textContent.trim().replace(/[%,+\s]/g, '')); return isNaN(n) ? null : n; };
        var rows = Array.prototype.slice.call(body.rows);
        var refs = rows.filter(function(r){ return r.classList.contains('ref'); });
        var data = rows.filter(function(r){ return !r.classList.contains('ref'); });
        data.sort(function(a, b){
          var ka = num(a.cells[idx]), kb = num(b.cells[idx]);
          if (ka === null && kb === null){ var c = a.cells[idx].textContent.trim().localeCompare(b.cells[idx].textContent.trim()); return asc ? c : -c; }
          if (ka === null) return 1;
          if (kb === null) return -1;
          return asc ? ka - kb : kb - ka;
        });
        refs.concat(data).forEach(function(r){ body.appendChild(r); });
      });
    });
  }
  document.querySelectorAll('table.sortable').forEach(makeSortable);

  var box = document.createElement('div'); box.id = 'tipbox'; document.body.appendChild(box);
  function show(el){
    var t = el.getAttribute('data-tip'); if (!t) return;
    box.textContent = t; box.style.display = 'block';
    var r = el.getBoundingClientRect();
    box.style.left = Math.max(6, Math.min(r.left, window.innerWidth - 252)) + 'px';
    var top = r.top - box.offsetHeight - 6; if (top < 6) top = r.bottom + 6;
    box.style.top = top + 'px';
  }
  function hide(){ box.style.display = 'none'; }
  document.addEventListener('mouseover', function(e){ var el = e.target.closest && e.target.closest('.tip'); if (el) show(el); });
  document.addEventListener('mouseout', function(e){ if (e.target.closest && e.target.closest('.tip')) hide(); });
  document.addEventListener('focusin', function(e){ var el = e.target.closest && e.target.closest('.tip'); if (el) show(el); });
  document.addEventListener('focusout', hide);
})();
"""


def _nav(active):
    links = "\n".join(
        f'  <a href="{f}"{" class=\"active\"" if f == active else ""}>{lbl}</a>'
        for f, lbl in TABS)
    return f'<nav class="tabs">\n{links}\n</nav>'


def page(title, active, body, extra_css="", extra_head=""):
    """The full HTML document: head (style.css link + shared CSS + extra_css),
    nav tabs with `active` (a TABS filename) highlighted, the .wrap'd body, and
    one copy of the shared JS. Page-specific <script>/<style> for the body can
    also be embedded in `body` directly — it runs before the shared JS."""
    style = SHARED_CSS + (extra_css or "")
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="style.css?v=22">
<style>{style}</style>
{extra_head}
</head><body>
{_nav(active)}
<div class="wrap">
{body}
</div>
<script>{SHARED_JS}</script>
</body></html>"""
