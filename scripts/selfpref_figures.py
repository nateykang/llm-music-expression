#!/usr/bin/env python3
"""Figures for the Part 2 self-preference page, from docs/analysis/selfpref_v3.json.

Palette: dataviz reference instance (blue sequential ramp; blue/red diverging
poles; chart chrome inks). Static PNGs into docs/analysis/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "docs/analysis/selfpref_v3.json").read_text())
J, ORDER = D["judges"], D["judges_order"]
BLUE, RED, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"
BLUE_LIGHT, BLUE_DARK = "#86b6ef", "#1c5cab"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def strip(ax):
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---- Fig 1: corrected strict-self bias per judge, diverging bars + 95% CI ----
rows = [(j, J[j]["strict_self"]) for j in ORDER if J[j]["strict_self"]]
rows.sort(key=lambda r: r[1]["corrected"])
fig, ax = plt.subplots(figsize=(7.2, 10.5), dpi=160)
ys = range(len(rows))
for y, (j, s) in zip(ys, rows):
    c = s["corrected"]; lo, hi = s["ci95"]
    ax.barh(y, c, height=0.62, color=BLUE if c >= 0 else RED, linewidth=0)
    ax.plot([lo, hi], [y, y], color=INK2, linewidth=1.2, solid_capstyle="butt")
ax.axvline(0, color=AXIS, linewidth=1)
ax.set_yticks(list(ys)); ax.set_yticklabels([j for j, _ in rows])
ax.set_xlabel("leniency-corrected self-preference (own pieces − others' pieces, quality scale points)")
ax.set_title("Does each model rate its own music higher than the panel does?", loc="left", color=INK, fontsize=11)
ax.text(0, 1.012, "blue = favors its own pieces · red = harsher on its own pieces · whiskers = 95% bootstrap CI over pieces",
        transform=ax.transAxes, color=INK2, fontsize=8)
strip(ax); ax.grid(axis="y", visible=False)
ax.set_xlim(-0.45, 0.85)
fig.tight_layout(); fig.savefig(ROOT / "docs/analysis/selfpref_bias.png"); plt.close(fig)

# ---- Fig 2: thinking vs base per family (dumbbell) ----
fam = {}
for j in ORDER:
    s = J[j]["strict_self"]
    if s: fam.setdefault(J[j]["family"], {})["thinking" if J[j]["thinking"] else "base"] = s["corrected"]
pairs = sorted([(f, v["base"], v["thinking"]) for f, v in fam.items() if "base" in v and "thinking" in v],
               key=lambda x: x[2] - x[1])
fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=160)
for y, (f, b, t) in enumerate(pairs):
    ax.plot([b, t], [y, y], color=GRID, linewidth=2, zorder=1)
    ax.scatter([b], [y], s=52, color=BLUE_LIGHT, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.scatter([t], [y], s=52, color=BLUE_DARK, edgecolor=SURFACE, linewidth=1.5, zorder=3)
ax.axvline(0, color=AXIS, linewidth=1)
ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p[0] for p in pairs])
ax.scatter([], [], s=52, color=BLUE_LIGHT, label="base arm"); ax.scatter([], [], s=52, color=BLUE_DARK, label="thinking arm")
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.set_xlabel("corrected self-preference")
ax.set_title("Same model, thinking on vs off: does reasoning change self-preference?", loc="left", color=INK, fontsize=11)
ax.text(0, 1.015, "sorted by thinking − base; families where thinking widened the self-preference gap sit at the bottom",
        transform=ax.transAxes, color=INK2, fontsize=8)
strip(ax); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(ROOT / "docs/analysis/selfpref_thinking.png"); plt.close(fig)

# ---- Fig 3: competence vs self-bias (scatter, single series, label extremes) ----
pts = [(J[j]["competence_r"], J[j]["strict_self"]["corrected"], j) for j in ORDER if J[j]["strict_self"]]
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=48, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3)
ax.axhline(0, color=AXIS, linewidth=1)
extremes = sorted(pts, key=lambda p: p[1])[:2] + sorted(pts, key=lambda p: p[1])[-3:] + sorted(pts, key=lambda p: p[0])[:2]
seen = set()
for x, y, j in extremes:
    if j in seen: continue
    seen.add(j); ax.annotate(j, (x, y), xytext=(6, 4), textcoords="offset points", color=INK2, fontsize=8)
r = D["summary"]["competence_vs_selfbias_r"]
ax.set_xlabel("judge competence: r(own scores, leave-one-out panel mean)")
ax.set_ylabel("corrected self-preference")
ax.set_title(f"Less panel-aligned judges favor themselves more (r = {r:+.2f}, n = {len(pts)})", loc="left", color=INK, fontsize=11)
ax.grid(color=GRID, linewidth=1); ax.set_axisbelow(True); ax.tick_params(length=0)
fig.tight_layout(); fig.savefig(ROOT / "docs/analysis/selfpref_competence.png"); plt.close(fig)
print("wrote selfpref_bias.png, selfpref_thinking.png, selfpref_competence.png")
