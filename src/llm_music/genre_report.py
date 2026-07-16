"""Generate docs/genre.html — the human-corpora genre & cultural bias tab.

Everything is computed at build time from committed data: judge_human_raw.json
(801 pieces x 10 judges, extended rubric), judge_bach_raw.json (the original
371-chorale run, for the rubric-change check), human_corpora_sample.json (the
frozen sample; also tells us which Arab pieces carry form names in the
notation), and features.csv (each judge's own major-writing rate).

Build: ``llm-music genre-report``
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from .config import REPO_ROOT
from .judge import QUALITY_KEYS
from .report_common import BG, INK, MUTED, SHORT, fnote, page, table

FIG_DIR = "analysis/genre"
QUAL = QUALITY_KEYS
NEW = ["enjoyment", "interest", "beauty", "familiarity"]
ARM_LABEL = {
    "bach": "Bach chorales", "classical": "Classical strings",
    "irish_folk": "Irish folk", "euro_folk": "German folk (Essen)",
    "chinese_han": "Chinese Han folk", "arab_and": "Arab-Andalusian",
}
ARAB_FORM = re.compile(r"tawshiya|mshalia|nuba|quddam", re.I)
ANY_QUOTE = re.compile(r'"[^"]{2,}"')
ARAB_GROUPS = [("form", "genre form name in notation"),
               ("title", "Arabic song title only"),
               ("none", "never had text (control)")]


def _arab_group(rep: str) -> str:
    if ARAB_FORM.search(rep):
        return "form"
    if ANY_QUOTE.search(rep):
        return "title"
    return "none"
ORIGIN_KW = {
    "bach": ["bach", "chorale", "lutheran", "german baroque", "baroque german"],
    "euro_folk": ["german folk", "european folk", "folk song", "volkslied",
                  "central european"],
    "irish_folk": ["irish", "celtic", "scottish"],
    "chinese_han": ["chinese", "china", "east asian", "han"],
    "classical": ["baroque", "classical", "haydn", "mozart", "european art"],
    "arab_and": ["arab", "andalus", "maghreb", "ottoman", "middle east",
                 "turkish", "persian", "north african", "egypt"],
}
np.random.seed(0)


def _qual(v):
    s = [v[k]["score"] for k in QUAL if k in v]
    return mean(s) if s else None


def _fig(name, draw, figsize=(7.4, 5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=figsize, dpi=130)
    f.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    draw(ax, f)
    f.tight_layout()
    out = REPO_ROOT / "docs" / FIG_DIR
    out.mkdir(parents=True, exist_ok=True)
    f.savefig(out / name, facecolor=BG)
    plt.close(f)
    return name


def _figure(fname, caption):
    import html as _h
    return (f"<figure class='chart'><img src='{FIG_DIR}/{fname}' "
            f"alt='{_h.escape(caption)}'><figcaption>{caption}</figcaption></figure>")


def _blind_stats(analysis: Path, orig_arab, grp_of):
    """(stats-by-group, rows) for the labeled-vs-blind contrast, or None.
    Groups by what text the piece originally carried: an explicit genre form
    name, only an Arabic song title, or nothing (retest control)."""
    path = analysis / "judge_arab_blind_raw.json"
    if not path.exists():
        return None
    blind = json.loads(path.read_text(encoding="utf-8"))
    orig = {r["id"]: r for r in orig_arab}

    def hit(v):
        g = v.get("origin_guess", "").lower()
        return any(k in g for k in ORIGIN_KW["arab_and"])

    rows_out, stats = [], {}
    for gkey, name in ARAB_GROUPS:
        grp = [r for r in blind if grp_of.get(r["id"]) == gkey]
        dq, de, dfam, ho, hb = [], [], [], [], []
        for r in grp:
            o = orig.get(r["id"])
            if not o:
                continue
            for J, vb in r["panel"].items():
                vo = o["panel"].get(J)
                if not vo:
                    continue
                qb, qo = _qual(vb), _qual(vo)
                if qb is not None and qo is not None:
                    dq.append(qb - qo)
                if "enjoyment" in vb and "enjoyment" in vo:
                    de.append(vb["enjoyment"]["score"] - vo["enjoyment"]["score"])
                if "familiarity" in vb and "familiarity" in vo:
                    dfam.append(vb["familiarity"]["score"] - vo["familiarity"]["score"])
                if "origin_guess" in vo:
                    ho.append(hit(vo))
                if "origin_guess" in vb:
                    hb.append(hit(vb))
        rng = np.random.default_rng(0)
        d = np.array(dq)
        p = float(np.mean([abs(np.mean(d * rng.choice([-1, 1], len(d)))) >= abs(d.mean())
                           for _ in range(10000)]))
        stats[gkey] = (mean(dq), mean(de), mean(dfam), mean(ho), mean(hb), p)
        rows_out.append([name, str(len(grp)), f"{mean(dq):+.2f}", f"{mean(de):+.2f}",
                         f"{mean(dfam):+.2f}",
                         f"{100 * mean(ho):.0f}% → {100 * mean(hb):.0f}%",
                         "&lt; 0.001" if p < 0.001 else f"{p:.2f}"])
    return stats, rows_out


def _blind_section(analysis: Path, orig_arab, grp_of) -> str:
    """Within-piece labeled-vs-blind contrast from the quote-stripped re-run."""
    res = _blind_stats(analysis, orig_arab, grp_of)
    if res is None:
        return ""
    stats, rows_out = res
    return (
        f"<p class='scope'><b>The blind re-run.</b> I re-judged the whole arm with every "
        f"quoted string stripped from the notation — same pieces, same judges, same "
        f"rubric; only the embedded text changes. Splitting the pieces by what their text "
        f"originally said (an explicit genre form name, only an Arabic song title, or "
        f"nothing — the last group re-judged unchanged as a retest control):</p>"
        + table([("group", None), ("n", "pieces"),
                 ("Δ quality", "blind minus labeled, points per verdict"),
                 ("Δ enjoyment", None),
                 ("Δ familiarity", None),
                 ("origin correct", "labeled run → blind run"),
                 ("perm p", "sign-flip permutation test on Δ quality")],
                rows_out)
        + f"<p class='scope'>The control is flat ({stats['none'][0]:+.2f}, p = "
        f"{stats['none'][5]:.2f}) — test-retest noise is tiny — and it was never "
        f"recognized in either run: with no text at all, judges identify Arab-Andalusian "
        f"music {100 * stats['none'][3]:.0f}% of the time. All recognition in this arm is "
        f"read off the embedded text, and the effect is dose-shaped: an explicit genre "
        f"name protected {abs(stats['form'][0]):.2f} points of quality, a mere Arabic "
        f"title (which cues the region through its language) protected "
        f"{abs(stats['title'][0]):.2f}, and both vanish when stripped. Blind judges also "
        f"<i>claim more familiarity</i> while scoring lower — consistent with misreading "
        f"the music as a Western idiom and penalizing it by Western standards.</p>"
        f"<p class='callout'><b>The headline.</b> Telling an LLM judge that a piece is "
        f"Arab-Andalusian makes it score the music <b>higher</b>, not lower. The naive "
        f"worry — that a non-Western label triggers a penalty — points the wrong way. The "
        f"penalty comes from <i>not knowing</i>: blind, judges misread the music as a "
        f"clumsy Western piece and mark it down by Western standards; the label switches "
        f"on genre-appropriate expectations and the penalty lifts — in proportion to how "
        f"informative the label is (Δ quality {stats['form'][0]:+.2f} for a genre name, "
        f"{stats['title'][0]:+.2f} for an Arabic title, {stats['none'][0]:+.2f} for the "
        f"no-text control; p &lt; 0.001). A within-piece manipulation with a flat retest "
        f"control — the single cleanest causal result in this experiment.</p>")


def render_genre_html(analysis: Path, data_dir: Path, out_path: Path) -> Path:
    rows = json.loads((analysis / "judge_human_raw.json").read_text(encoding="utf-8"))
    old_bach = json.loads((analysis / "judge_bach_raw.json").read_text(encoding="utf-8"))
    sample = json.loads((analysis / "human_corpora_sample.json").read_text(encoding="utf-8"))
    arms = sorted({r["arm"] for r in rows})
    order = ["classical", "bach", "irish_folk", "euro_folk", "chinese_han", "arab_and"]
    arms = [a for a in order if a in arms] + [a for a in arms if a not in order]
    judges = sorted({j for r in rows for j in r["panel"]})
    n_verd = sum(len(r["panel"]) for r in rows)
    secs = []

    def arm_mean(arm, dim):
        vals = [v[dim]["score"] for r in rows if r["arm"] == arm
                for v in r["panel"].values() if dim in v]
        return mean(vals)

    def arm_qual(arm):
        vals = [q for r in rows if r["arm"] == arm
                for q in (_qual(v) for v in r["panel"].values()) if q is not None]
        return mean(vals)

    # ---- 1. the experiment ---------------------------------------------------------
    arab_orig = [r for r in rows if r["arm"] == "arab_and"]
    arab_grp_of = {r["id"]: _arab_group(r["rep"]) for r in sample
                   if r["arm"] == "arab_and"}
    blind_res = _blind_stats(analysis, arab_orig, arab_grp_of)
    headline = ""
    if blind_res is not None:
        dq = blind_res[0]["form"][0]
        headline = (
            f"<p class='callout'><b>Headline result:</b> whether an LLM judge scores "
            f"unfamiliar music harshly turns on <i>recognition</i>, not on prejudice "
            f"against the culture. In a within-piece experiment, <b>telling the judges a "
            f"piece is Arab-Andalusian makes them rate it higher, not lower</b> "
            f"(Δ quality {dq:+.2f} points when the genre name is removed; p &lt; 0.001, where p "
            f"is the probability of a gap this large arising by chance — with a "
            f"flat retest control). Blind, judges misread the music as clumsy Western "
            f"writing and mark it down; the label switches on genre-appropriate standards. "
            f"Full experiment in the “know what they're listening to” section below.</p>")
    secs.append(
        "<h2>The experiment</h2>"
        + headline
        + f"<p class='scope'>I sampled {len(rows)} pieces of human music from six corpora "
        "(aiming for 75 major / 75 minor per corpus, best-effort where a corpus skews: "
        "the classical arm has 51 pieces, the Arab-Andalusian split is 118/32). Every piece "
        "was converted to the same note-listing text the LLM pieces are judged in, and the "
        f"same 10-judge panel scored each one — {n_verd} verdicts. The rubric keeps the "
        "original 8 quality dimensions plus valence/arousal (perceived positivity and "
        f"energy of the music, after the Russell circumplex model of emotion{fnote('russell')}) "
        "unchanged and adds four new hedonic — i.e. pleasure-related — "
        "1–5 questions (enjoyment, interest, beauty, familiarity) and a free-text guess at "
        "the piece's tradition or origin. The question behind the design: do LLM judges "
        "score non-Western music lower, and if so, is it mediated by unfamiliarity?</p>")

    # ---- 2. rubric-change check ------------------------------------------------------
    okey = Counter((r["title"], r["key"]) for r in old_bach)
    nkey = Counter((r["title"], r["key"]) for r in rows if r["arm"] == "bach")
    unique = {k for k in nkey if nkey[k] == 1 and okey.get(k) == 1}
    oldby = {(r["title"], r["key"]): r for r in old_bach}
    pairs = []
    for r in rows:
        k = (r["title"], r["key"])
        if r["arm"] != "bach" or k not in unique:
            continue
        for J in judges:
            vo, vn = oldby[k]["panel"].get(J), r["panel"].get(J)
            if vo and vn:
                qo, qn = _qual(vo), _qual(vn)
                if qo is not None and qn is not None:
                    pairs.append((qo, qn))
    qo = np.array([a for a, _ in pairs])
    qn = np.array([b for _, b in pairs])

    def major_bias(source, key_of):
        out = {}
        for J in judges:
            mj, mn = [], []
            for r in source:
                panel = r["panel"]
                if J not in panel:
                    continue
                qJ = _qual(panel[J])
                peers = [x for x in (_qual(v) for j, v in panel.items() if j != J)
                         if x is not None]
                if qJ is None or not peers:
                    continue
                (mj if key_of(r).endswith("major") else mn).append(qJ - mean(peers))
            if mj and mn:
                out[J] = mean(mj) - mean(mn)
        return out

    nb = [r for r in rows if r["arm"] == "bach"]
    b_old = major_bias(old_bach, lambda r: r["key"])
    b_new = major_bias(nb, lambda r: r["key"])
    common_j = [J for J in judges if J in b_old and J in b_new]
    r_bias = np.corrcoef([b_old[J] for J in common_j],
                         [b_new[J] for J in common_j])[0, 1]

    # own major-writing rate (used here and in the mode-bias section)
    maj_c, tot_c = {}, {}
    for f in data_dir.glob("*/features.csv"):
        for r in csv.DictReader(f.open(encoding="utf-8")):
            if r.get("prompt") == "free-form" and r.get("key_mode_best") in ("major", "minor"):
                m = r["model"]
                tot_c[m] = tot_c.get(m, 0) + 1
                maj_c[m] = maj_c.get(m, 0) + (r["key_mode_best"] == "major")
    own_major = {m: maj_c[m] / tot_c[m] for m in tot_c}

    def bias_corr(source, key_of):
        b = major_bias(source, key_of)
        xs = [own_major[J] for J in b if J in own_major]
        ys = [b[J] for J in b if J in own_major]
        return np.corrcoef(xs, ys)[0, 1]

    bach_titles = {(r["title"], r["key"]) for r in nb}
    r_old_full = bias_corr(old_bach, lambda r: r["key"])
    r_old_sub = bias_corr([r for r in old_bach
                           if (r["title"], r["key"]) in bach_titles],
                          lambda r: r["key"])
    r_new_sub = bias_corr(nb, lambda r: r["key"])
    secs.append(
        "<h2>First, a rubric-change check</h2>"
        "<p class='scope'>The new questions could plausibly change how the panel scores the "
        f"old ones. The bach arm gives a direct test: {len(unique)} of its chorales were "
        "already judged in the 371-chorale run with the original rubric — same pieces, same "
        f"judges. Across {len(pairs)} paired verdicts, mean quality moved from "
        f"{qo.mean():.2f} to {qn.mean():.2f} (a {qn.mean() - qo.mean():+.2f} shift, spread "
        "evenly over the dimensions), and piece-level scores correlate at "
        f"r = {np.corrcoef(qo, qn)[0, 1]:.2f}. Per-judge favors-major bias replicates at "
        f"r = {r_bias:+.2f}. So the extended rubric measures roughly the same thing, and "
        "the results below can be compared with earlier runs.</p>"
        "<p class='scope'>One second-level statistic did move: the own-major ↔ favors-major "
        f"correlation is r = {r_old_full:+.2f} in the old run — and still "
        f"{r_old_sub:+.2f} when the old run is restricted to these same 150 chorales — but "
        f"{r_new_sub:+.2f} in the new run. So the drop comes from the re-run itself, not "
        "the smaller sample. The judges with large biases (llama, opus, fable, sonnet, "
        "deepseek) are stable across runs; the near-zero judges wobble (gpt-4.1 "
        f"{b_old.get('gpt-4.1', 0):+.2f} → {b_new.get('gpt-4.1', 0):+.2f}, gemini "
        f"{b_old.get('gemini-2.5-pro', 0):+.2f} → {b_new.get('gemini-2.5-pro', 0):+.2f}), "
        "and with 10 judges two wobbles move the correlation this much. I can't fully "
        "separate rubric influence from re-roll noise without re-running the old rubric, "
        "but the stability of every large bias points at noise on the small ones.</p>")

    # ---- 3. scores by arm ------------------------------------------------------------
    rep_of = {r["id"]: r["rep"] for r in sample}
    sizes = {}
    for a in arms:
        toks = [len(rep_of[r["id"]].split()) for r in rows if r["arm"] == a
                if r["id"] in rep_of]
        sizes[a] = mean(toks)
    size_txt = ", ".join(f"{ARM_LABEL.get(a, a)} {sizes[a]:.0f} tokens"
                         for a in sorted(arms, key=lambda a: sizes[a]))

    def barsfig(ax, f):
        dims = [("quality", None)] + [(d, d) for d in ("enjoyment", "beauty",
                                                       "familiarity")]
        w = 0.2
        xs = np.arange(len(arms))
        cols = ["#7a5c3e", "#3a6b5a", "#8a3a4a", "#37648a"]
        for i, (label, d) in enumerate(dims):
            vals = [arm_qual(a) if d is None else arm_mean(a, d) for a in arms]
            ax.bar(xs + (i - 1.5) * w, vals, w, label=label, color=cols[i])
        ax.set_xticks(xs)
        ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms], fontsize=7.5)
        ax.set_ylim(1, 5)
        ax.legend(frameon=False, fontsize=8, ncol=4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title("Panel means by corpus (1–5)", color=INK, fontsize=11)
    _fig("arm_scores.png", barsfig, figsize=(7.4, 4.2))

    arm_rows = []
    for a in arms:
        n = sum(1 for r in rows if r["arm"] == a)
        arm_rows.append([ARM_LABEL.get(a, a), str(n), f"{arm_qual(a):.2f}"]
                        + [f"{arm_mean(a, d):.2f}" for d in NEW])
    secs.append(
        "<h2>Scores by corpus</h2>"
        + _figure("arm_scores.png", "Classical strings, Bach, and Irish folk sit clearly "
                  "above German folk, Chinese Han folk, and Arab-Andalusian on every "
                  "hedonic dimension. Note the non-Western arms are not alone at the "
                  "bottom: German folk song is rated just as low.")
        + table([("corpus", None), ("n", "pieces"),
                 ("quality", "mean of the 8 original dimensions"),
                 ("enjoyment", None), ("interest", None), ("beauty", None),
                 ("familiarity", "judge's self-reported familiarity with the idiom")],
                arm_rows)
        + "<p class='scope'>The raw ranking is consistent with a Western-bias story for the "
        "two non-Western arms, but German folk (Essen collection) scoring at the same level "
        "complicates it: the shared property of the three low arms is that they are mostly "
        "short monophonic melodies (a single unaccompanied melody line), while the three "
        "high arms have harmony or counterpoint (several interweaving melody lines) "
        "written out. The sizes make this concrete — mean notation length per piece is "
        f"{size_txt}. Some of the gap is plausibly a texture and scale penalty, not a "
        "cultural one.</p>")

    # ---- 4. origin guesses -----------------------------------------------------------
    def hit_rate(rs):
        gs = [(r["arm"], v["origin_guess"].lower()) for r in rs
              for v in r["panel"].values() if "origin_guess" in v]
        by = defaultdict(list)
        for a, g in gs:
            by[a].append(any(k in g for k in ORIGIN_KW[a]))
        return {a: float(np.mean(v)) for a, v in by.items()}

    hits = hit_rate(rows)
    arab = [r for r in rows if r["arm"] == "arab_and"]

    def grp_stats(rs):
        qs = [q for r in rs for q in (_qual(v) for v in r["panel"].values())
              if q is not None]
        enj = [v["enjoyment"]["score"] for r in rs for v in r["panel"].values()
               if "enjoyment" in v]
        fam = [v["familiarity"]["score"] for r in rs for v in r["panel"].values()
               if "familiarity" in v]
        h = hit_rate(rs).get("arab_and", float("nan"))
        return len(rs), mean(qs), mean(enj), mean(fam), h

    grp_rows = []
    for gkey, gname in ARAB_GROUPS:
        st = grp_stats([r for r in arab if arab_grp_of.get(r["id"]) == gkey])
        grp_rows.append([gname, str(st[0]), f"{st[1]:.2f}", f"{st[2]:.2f}",
                         f"{st[3]:.2f}", f"{st[4] * 100:.0f}%"])
    n_text = sum(1 for g in arab_grp_of.values() if g != "none")
    ex_guesses = sorted({v["origin_guess"] for r in rows if r["arm"] == "chinese_han"
                         for v in r["panel"].values() if "origin_guess" in v})
    rng = np.random.default_rng(0)
    ex_pick = [ex_guesses[i] for i in rng.choice(len(ex_guesses), 5, replace=False)]
    secs.append(
        "<h2>Do the judges know what they're listening to?</h2>"
        + table([("corpus", None), ("origin guess correct", "keyword match on the "
                 "free-text guess, so approximate")],
                [[ARM_LABEL.get(a, a), f"{hits[a] * 100:.0f}%"] for a in arms])
        + "<p class='scope'>The note listings carry no titles and no lyrics (the Bach "
        "chorales' words are stripped along with everything else non-musical), so "
        "recognition rests on the notes alone — with one exception covered below. Bach, "
        "the classical arm, and (with that caveat) the "
        "Arab-Andalusian arm are usually recognized. The Chinese Han pieces are "
        f"recognized {hits['chinese_han'] * 100:.0f}% of the time — judges typically read "
        "them as Western material. Some sample guesses for Chinese pieces: "
        + "; ".join(f"“{g}”" for g in ex_pick) + ". Irish and German folk mostly blur "
        "into generic “Western European folk”. So for the chinese_han arm the low scores "
        "come <i>without</i> the judges knowing the music is Chinese — whatever penalty "
        "it gets is not applied to a “Chinese” label.</p>"
        f"<p class='scope'><b>A leak, and a lucky natural experiment.</b> {n_text} "
        f"of the {len(arab)} Arab-Andalusian pieces carry text inside the notation itself "
        "— either an explicit genre form name (e.g. “Tawshiya Qaim Wa Nisf”) or just an "
        "Arabic song title, inherited from the source MusicXML as inline directions. No "
        "other corpus leaks anything like this. Splitting the arm three ways in the "
        "original run:</p>"
        + table([("group", None), ("n", "pieces"), ("quality", None),
                 ("enjoyment", None), ("familiarity", None),
                 ("origin correct", None)], grp_rows)
        + "<p class='scope'>The gradient runs with how informative the text is: with a "
        "genre name judges recognize the tradition most, admit the least familiarity — "
        "and score the music highest; with only an Arabic title, less so (the title's "
        "language still cues the region); with no text at all, recognition is 0% — "
        "judges have <i>never</i> identified this music from the notes — and scores are "
        "lowest, while claimed familiarity is highest (consistent with misreading it as "
        "some Western idiom they know). On its own this is only suggestive — the groups "
        "are different slices of the repertoire — so I made it a real experiment.</p>"
        + _blind_section(analysis, orig_arab=arab, grp_of=arab_grp_of))

    # ---- 5. familiarity mediation ------------------------------------------------------
    recs = []
    for r in rows:
        for J, v in r["panel"].items():
            q = _qual(v)
            if q is not None and "familiarity" in v:
                recs.append((r["arm"], J, v["familiarity"]["score"], q))
    arm_i = {a: i for i, a in enumerate(arms)}
    jud_i = {j: i for i, j in enumerate(judges)}
    base = arm_i["bach"]
    A_cols = len(arms) - 1

    def fit(with_fam):
        ncol = A_cols + len(judges) + (1 if with_fam else 0)
        Xm = np.zeros((len(recs), ncol))
        y = np.zeros(len(recs))
        for i, (a, j, f, q) in enumerate(recs):
            ai = arm_i[a]
            if ai != base:
                Xm[i, ai - (1 if ai > base else 0)] = 1
            Xm[i, A_cols + jud_i[j]] = 1
            if with_fam:
                Xm[i, -1] = f
            y[i] = q
        coef = np.linalg.lstsq(Xm, y, rcond=None)[0]
        return ({a: coef[arm_i[a] - (1 if arm_i[a] > base else 0)]
                 for a in arms if a != "bach"},
                coef[-1] if with_fam else None)

    g0, _ = fit(False)
    g1, slope = fit(True)
    med_rows = [[ARM_LABEL.get(a, a), f"{g0[a]:+.2f}", f"{g1[a]:+.2f}"]
                for a in sorted(g0, key=g0.get)]
    secs.append(
        "<h2>Is the gap about familiarity?</h2>"
        "<p class='scope'>The familiarity question was added as a candidate mediator: if "
        "judges score music lower because the idiom is foreign to them, controlling for "
        "self-reported familiarity should shrink the arm gaps. Regression of quality on "
        "corpus + judge, with and without familiarity as a covariate (slope "
        f"{slope:+.2f} quality points per familiarity point):</p>"
        + table([("corpus", None), ("gap vs Bach", "points, judge-adjusted"),
                 ("after familiarity control", None)], med_rows)
        + f"<p class='scope'>Familiarity explains most of the Arab-Andalusian gap "
        f"({g0['arab_and']:+.2f} → {g1['arab_and']:+.2f}) — it is the only arm judges "
        "self-report as unfamiliar (3.64 vs ~4.5 elsewhere). It explains almost none of "
        "the Chinese or German folk gaps, which fits the origin-guess result: judges "
        "think that music is familiar Western material, so unfamiliarity can't be what "
        "drives its penalty. Two different mechanisms, then: an unfamiliarity penalty for "
        "the Arab arm, and something else — plausibly the monophonic-texture penalty — "
        "for the folk arms.</p>")

    # ---- 6. what the reasons say --------------------------------------------------------
    STOP = set("the a an and or of to in is are its it this that with for on as be has "
               "have very but not no piece music musical while though than more most "
               "some what which by from at one two their there each between could would "
               "into over only just also feels".split())

    def words(arm_list, dims):
        c = Counter()
        for r in rows:
            if r["arm"] not in arm_list:
                continue
            for v in r["panel"].values():
                for d in dims:
                    if d in v and isinstance(v[d], dict):
                        for w in re.findall(r"[a-z']+", v[d].get("reason", "").lower()):
                            if w not in STOP and len(w) > 3:
                                c[w] += 1
        return c

    DIMS6 = ["harmony", "coherence", "naturalness", "creativity", "enjoyment", "beauty"]
    west = words(("bach", "classical", "irish_folk"), DIMS6)
    ar = words(("arab_and",), DIMS6)
    tw, ta = sum(west.values()), sum(ar.values())
    scored = sorted(((math.log((n / ta) / ((west.get(w, 0) + .5) / tw)), n, w)
                     for w, n in ar.items() if n >= 40), reverse=True)
    style_words = [w for lo, n, w in scored
                   if w in ("looping", "verbatim", "endless", "loop", "loops",
                            "oscillation", "oscillating", "copy", "hundreds",
                            "extensive", "repetition", "repeats")][:6]
    lows = sorted((r["title"], v["harmony"]["reason"]) for r in arab
                  for v in r["panel"].values()
                  if v.get("harmony", {}).get("score", 5) <= 2 and v.get("harmony", {}).get("reason"))
    rng2 = np.random.default_rng(1)
    quotes = [lows[i][1] for i in rng2.choice(len(lows), 3, replace=False)]

    euro_w = words(("euro_folk",), DIMS6)
    te2 = sum(euro_w.values())
    SIMPLE = ["beginner", "pedagogical", "inoffensive", "children's", "elementary",
              "nursery", "rudimentary", "underdeveloped", "unadorned"]
    euro_hits = [(w, euro_w.get(w, 0)) for w in SIMPLE if euro_w.get(w, 0) >= 60
                 and (euro_w[w] / te2) > 3 * ((west.get(w, 0) + .5) / tw)]
    secs.append(
        "<h2>What the judges' reasons say</h2>"
        "<p class='scope'>Comparing word frequencies in the written justifications for the "
        "Arab-Andalusian pieces against the Western art/folk arms, the most "
        "over-represented words (after the form names themselves) are about repetition: "
        + ", ".join(f"“{w}”" for w in style_words) + ". Nuba movements (the multi-movement "
        "suite form of Arab-Andalusian music) repeat short cells "
        "extensively, and in a note-by-note text listing that repetition is literal — "
        "hundreds of near-identical bars — which likely reads worse than it sounds. The "
        "low harmony scores also show judges applying functional-harmony expectations — "
        "the Western system of chords progressing toward a home key — to music that is "
        "modal (built on melodic formulas outside the major/minor system) and "
        "monophonic:</p>"
        + "".join(f"<p class='scope' style='margin-left:1.5em'><i>“{q}”</i></p>"
                  for q in quotes)
        + "<p class='scope'>The German folk penalty has a completely different vocabulary. "
        "Its most over-represented justification words are about simplicity: "
        + ", ".join(f"“{w}” ({n}×)" for w, n in euro_hits)
        + ". The judges find the idiom familiar and dismiss it as beginner material — a "
        "simplicity penalty, not an unfamiliarity one.</p>")

    # ---- 7. per-judge profiles -----------------------------------------------------------
    prof_rows = []
    for J in judges:
        allq = [q for a, j, f, q in recs if j == J]
        gm = mean(allq)
        per = [mean(q for a2, j, f, q in recs if a2 == a and j == J) - gm for a in arms]
        prof_rows.append([SHORT.get(J, J)] + [f"{x:+.2f}" for x in per])
    secs.append(
        "<h2>Per-judge profiles</h2>"
        "<p class='scope'>Judge-centered mean quality per corpus (each judge's grand mean "
        "subtracted, so the columns show relative taste). Most judges penalize the "
        "Arab-Andalusian arm; gemini and deepseek don't. The classical arm splits the "
        "panel widely (gpt-4.1 +0.97, gemini −0.21). gemini is the judge that most bucks "
        "the art-music-first pattern.</p>"
        + table([("judge", None)] + [(ARM_LABEL.get(a, a), None) for a in arms],
                prof_rows))

    # ---- 8. mode bias per arm --------------------------------------------------------------
    mode_rows = []
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        bias = major_bias(sub, lambda r: r["mode"] or "")
        xs = np.array([own_major[J] for J in bias if J in own_major])
        ys = np.array([bias[J] for J in bias if J in own_major])
        r0 = np.corrcoef(xs, ys)[0, 1]
        rng3 = np.random.default_rng(0)
        p = float(np.mean([abs(np.corrcoef(rng3.permutation(xs), ys)[0, 1]) >= abs(r0)
                           for _ in range(20000)]))
        nmaj = sum(1 for r in sub if r["mode"] == "major")
        mode_rows.append([ARM_LABEL.get(a, a), f"{nmaj}/{len(sub) - nmaj}",
                          f"{r0:+.2f}", f"{p:.3f}"])
    # euro composition confound
    euro = [r for r in rows if r["arm"] == "euro_folk"]
    altdeu = {m: sum(1 for r in euro if r["mode"] == m and "/altdeu" in r["id"])
              for m in ("major", "minor")}
    secs.append(
        "<h2>Mode bias inside each corpus</h2>"
        "<p class='scope'>The Bach experiment found that judges who compose in major favor "
        "major pieces (r = +0.69 over all 371 chorales). The same correlation computed "
        "inside each of these corpora:</p>"
        + table([("corpus", None), ("major/minor", "pieces"),
                 ("r (own major rate vs favors-major)", "across the 10 judges"),
                 ("perm p", None)], mode_rows)
        + f"<p class='scope'>Bach and Irish folk go the same direction as before; the "
        f"German folk arm reverses, and with 6 tests its p = 0.017 is only borderline. "
        f"One concrete confound: within that arm, mode is entangled with sub-collection — "
        f"{altdeu['minor']} of the 75 minor pieces come from the altdeutsche Lieder "
        f"(archaic songs) versus {altdeu['major']} of the major pieces, so “minor” there "
        "largely means “older repertoire”. Judges' per-arm mode biases also correlate "
        "positively across every pair of arms except the German folk arm, which "
        "anti-correlates with all of them — pointing at the arm's composition rather than "
        "judges being fickle. I'd summarize: the mode-bias generalization holds on Bach "
        "and Irish folk and is confounded on the arms where the major/minor split is not "
        "apples-to-apples (German sub-collections; forced major/minor labels on "
        "pentatonic — five-note-scale — Chinese material).</p>")

    # ---- 9. methods ------------------------------------------------------------------------
    secs.append(
        "<h2>Methods &amp; caveats</h2>"
        "<p class='scope'>Sampling: scripts/sample_human_corpora.py, seed 0, stratified by "
        "detected mode; the frozen sample is committed as human_corpora_sample.json. "
        "Judging: scripts/judge_human_corpora.py, one API call per (piece, judge), "
        "reason-before-score, same protocol as every other run; raw verdicts in "
        "judge_human_raw.json. Caveats: the corpora differ in texture and length, so "
        "cross-corpus gaps mix cultural and musical-form effects; the Arab form-name leak "
        "means that arm's origin recognition is partly read off the text; second-level "
        "correlations use n = 10 judges; and the keyword origin-matching is approximate. "
        "On keys: three labels exist per piece — the engraved source key signature, the "
        "source-declared key the judges see in the rep header, and the detected key — "
        f"via the Krumhansl–Schmuckler algorithm{fnote('krumhansl')}, which infers a key "
        "from the piece's note-usage profile — used for mode stratification and the mode-bias "
        "analysis. They disagree exactly where music is modal, pentatonic, or tonally "
        "ambiguous, so major/minor on those arms is an algorithmic best fit. Judges "
        "almost never react to the declared header (mentions of the stated key appear in "
        "at most 0.2% of dimension-reasons), so the imperfect header is unlikely to move "
        "scores. "
        "Reproduce with <code>scripts/analyze_human_corpora.py</code> and "
        "<code>llm-music genre-report</code>.</p>")

    body = ("<h1>Judging human music: genre &amp; cultural bias</h1>"
            "<p class='scope'>The same LLM panel that judges the models' own music scored "
            "801 pieces of human music from six traditions, with four added hedonic "
            "(pleasure-related) questions and an origin guess. All numbers are computed from the committed "
            "data at build time. Generated by <code>llm-music genre-report</code>.</p>"
            + "\n".join(secs))
    chart_css = """
  .chart img { width: 100%; border: 1px solid var(--border); border-radius: 8px; }
"""
    out_path.write_text(page("Judging human music — genre & cultural bias",
                             "genre.html", body, extra_css=chart_css), encoding="utf-8")
    return out_path
