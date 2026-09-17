"""Part 1 pilot analysis: does a judge's rating of its OWN piece depend on key?

Per arm: within-piece-centered quality by shift and by TARGET pitch class
(tonic), flatness permutation test (shuffle shift labels within piece),
accidental-density regression, and the arm's generation key distribution for
the revealed-vs-expressed eyeball. Part 2 self verdicts on the same pieces
serve as the shift-0 reference point (cross-session, so shown but not tested).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
Q = ["coherence", "harmony", "rhythm", "structure", "melody", "emotion", "creativity", "naturalness"]
PC_OF = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
rng = np.random.default_rng(20260917)


def q8(v):
    return float(np.mean([v[k]["score"] for k in Q])) if v and all(k in v for k in Q) else None


def piece_tonic_pc(rep_text):
    m = re.match(r"Key(?: signature)?: ([A-G])([#-]?)", rep_text.splitlines()[0])
    if not m:
        return None
    return (PC_OF[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "-" else 0)) % 12


def main() -> int:
    ck = json.loads((ROOT / "part1_run/pilot_ckpt.json").read_text())
    reps = json.loads((ROOT / "part1_run/pilot_reps.json").read_text())
    man = json.loads((ROOT / "part1_run/pilot_manifest.json").read_text())
    dens = {k: (lambda t: len(re.findall(r"[A-G][#-]\d", t)) / max(1, len(re.findall(r"[A-G][#-]?\d", t))))(v)
            for k, v in reps.items()}
    # part 2 shift-0 self verdicts for the same pieces
    raw = json.loads((ROOT / "part2_run/analysis/part2_selfpref_raw.json").read_text())
    p2 = {(p["model"], p["sample"]): q8(p["panel"].get(p["model"])) for p in raw
          if p["mode"] == "codegen" and p["prompt"] == "express-yourself"}

    rows = defaultdict(list)  # arm -> [(sample, shift, q, density, tonic_pc)]
    n_ok = n_fail = 0
    for key, v in ck.items():
        batch, arm, prompt, mode, sample, shift, judge = key.rsplit("|", 6)[0], *key.split("|")[1:]
        arm, sample, shift = key.split("|")[1], int(key.split("|")[4]), int(key.split("|")[5])
        s = q8(v)
        if s is None:
            n_fail += 1
            continue
        n_ok += 1
        rk = key.rsplit("|", 1)[0]
        rows[arm].append((sample, shift, s, dens[rk], piece_tonic_pc(reps[rk])))
    print(f"verdicts: {n_ok} ok, {n_fail} failed of {sum(len(p) * 11 for p in man['picked'].values())} expected\n")

    out = {}
    shared = defaultdict(list)
    for arm in sorted(rows):
        R = rows[arm]
        by_piece = defaultdict(list)
        for smp, sh, s, d, pc in R:
            by_piece[smp].append((sh, s, d, pc))
        cent = []  # (shift, centered q, density, tonic pc)
        for smp, lst in by_piece.items():
            mu = np.mean([x[1] for x in lst])
            cent += [(sh, s - mu, d, pc) for sh, s, d, pc in lst]
        prof = {sh: float(np.mean([c for s_, c, *_ in cent if s_ == sh])) for sh in sorted({c[0] for c in cent})}
        spread = float(np.std(list(prof.values())))
        # permutation test: shuffle shifts within piece
        null = []
        for _ in range(2000):
            fake = []
            for smp, lst in by_piece.items():
                mu = np.mean([x[1] for x in lst])
                sh_perm = rng.permutation([x[0] for x in lst])
                fake += [(shp, s - mu) for shp, (sh, s, d, pc) in zip(sh_perm, lst)]
            fp = [np.mean([c for s_, c in fake if s_ == sh]) for sh in prof]
            null.append(np.std(fp))
        pval = float(np.mean([n >= spread for n in null]))
        # density slope on centered scores
        x = np.array([c[2] for c in cent]); y = np.array([c[1] for c in cent])
        slope = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else 0.0
        for sh, val in prof.items():
            shared[sh].append(val)
        out[arm] = {"profile": prof, "spread": spread, "p_flat": pval, "density_slope": slope,
                    "n": len(R), "p2_key0_mean": float(np.mean([p2.get((arm, s)) for s, *_ in R
                                                                if p2.get((arm, s)) is not None] or [np.nan]))}
    print(f"{'arm':26s} spread  p(flat)  dens.slope   best shift / worst shift")
    for arm, r in sorted(out.items(), key=lambda kv: kv[1]["p_flat"]):
        pr = r["profile"]; b = max(pr, key=pr.get); w = min(pr, key=pr.get)
        flag = " *" if r["p_flat"] < 0.05 else ""
        print(f"{arm:26s} {r['spread']:.3f}  {r['p_flat']:.3f}   {r['density_slope']:+.2f}      "
              f"{b:+d} ({pr[b]:+.2f}) / {w:+d} ({pr[w]:+.2f}){flag}")
    print(f"\narms with p<0.05 (non-flat own-key profile): {sum(1 for r in out.values() if r['p_flat'] < 0.05)}/{len(out)}")
    print("\nshared profile (mean centered q by shift, all arms):")
    for sh in sorted(shared):
        m = np.mean(shared[sh]); se = np.std(shared[sh]) / np.sqrt(len(shared[sh]))
        print(f"  {sh:+d}: {m:+.3f} ± {2*se:.3f}")
    (ROOT / "part1_run/pilot_analysis.json").write_text(json.dumps(out, indent=1))
    print("\nwrote part1_run/pilot_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
