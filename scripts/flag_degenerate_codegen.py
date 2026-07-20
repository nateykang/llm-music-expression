#!/usr/bin/env python3
"""Retroactively flag degenerate codegen pieces as failed generations.

A piece is degenerate when its exported MusicXML (the ground truth of what is
heard and engraved) has an instrument part with zero notes, or a silent tail
longer than max(4, 20%) of its measures — the same criteria the sandbox now
enforces at generation time (_sandbox_runner._validate_exported).

For each hit: sets ok=false (+ error, degenerate=true) in the batch's
data.json, and records it in docs/analysis/degenerate_pieces.json, which the
report builders use to exclude these pieces from judge/embedding analyses.
Idempotent; re-run after adding batches (new batches are protected by the
sandbox guard anyway).

    python scripts/flag_degenerate_codegen.py
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scan_xml(path: Path):
    root = ET.parse(path).getroot()
    names = {sp.get("id"): (sp.findtext("part-name") or sp.get("id"))
             for sp in root.iter("score-part")}
    empties, max_meas, sounding = [], 0, set()
    for part in root.iter("part"):
        notes = 0
        measures = part.findall("measure")
        max_meas = max(max_meas, len(measures))
        for mi, m in enumerate(measures):
            k = sum(1 for n in m.findall("note") if n.find("rest") is None)
            if k:
                notes += k
                sounding.add(mi)
        if notes == 0:
            empties.append(names.get(part.get("id"), part.get("id")))
    last = max(sounding) + 1 if sounding else 0
    return empties, max_meas, last


def main():
    flagged = []
    for dj in sorted((ROOT / "docs/data").glob("*/data.json")):
        batch = dj.parent
        d = json.loads(dj.read_text(encoding="utf-8"))
        changed = False
        for p in d["pieces"]:
            if not p.get("ok") or "codegen" not in (p.get("mode") or "") \
                    or not p.get("score"):
                continue
            xml = batch / p["score"]
            if not xml.exists():
                continue
            try:
                empties, nm, last = scan_xml(xml)
            except ET.ParseError:
                continue
            tail = nm - last
            reason = None
            if empties:
                reason = ("instrument part(s) with zero notes: "
                          + ", ".join(str(e) for e in empties))
            elif nm and tail > max(4, 0.2 * nm):
                reason = f"final {tail} of {nm} measures silent in every part"
            if not reason:
                continue
            p["ok"] = False
            p["degenerate"] = True
            p["error"] = f"degenerate score (flagged retroactively): {reason}"
            changed = True
            flagged.append({"model": p["model"], "mode": p["mode"],
                            "prompt": p["prompt"], "title": p.get("title", ""),
                            "sample": p.get("sample", 0), "batch": batch.name,
                            "reason": reason})
            print(f"flagged: {batch.name} {p['model']} {p['prompt']} "
                  f"s{p.get('sample', 0)} — {reason}")
        if changed:
            dj.write_text(json.dumps(d, indent=2, ensure_ascii=False),
                          encoding="utf-8")

    out = ROOT / "docs/analysis/degenerate_pieces.json"
    out.write_text(json.dumps(flagged, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n{len(flagged)} pieces flagged → {out}")

    # warn if a blacklist key collides with a healthy piece elsewhere
    # (raw-judge/embedding keys carry no batch, so a collision would over-drop)
    keys = Counter((f["model"], f["mode"], f["title"], str(f["sample"]))
                   for f in flagged)
    for dj in (ROOT / "docs/data").glob("*/data.json"):
        for p in json.loads(dj.read_text(encoding="utf-8"))["pieces"]:
            k = (p.get("model"), p.get("mode"), p.get("title", ""),
                 str(p.get("sample", 0)))
            if p.get("ok") and k in keys:
                print(f"WARNING: healthy piece shares key with a flagged one: {k} "
                      f"in {dj.parent.name} — it would be dropped too")


if __name__ == "__main__":
    main()
