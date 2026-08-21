# Judge-representation experiments (July–Aug 2026)

Two pre-registration-style protocol experiments on the LLM judge, run before the
v3 corpus. Scripts ran on the RunPod CPU against the v1 corpus; results live in
`docs/analysis/format_ab/`.

## 1. Format A/B: does the judge's input representation change ratings?

Historically, code-gen pieces were judged as a note-listing (`_score_to_text`)
while ABC pieces were judged as raw ABC text. `judge_format_ab.py` takes the
code-gen pieces whose MusicXML converts to ABC with an event-exact MIDI match
(134 pieces; `xml2abc_verify.json` holds the fidelity check over all 568 — 74%+
verified exact, every failure class diagnosed), and judges each piece BOTH ways
with the same 3-judge panel and the same prompt.

Findings (`judge_format_ab.json`):
- Overall score: converted-ABC − note-listing = **+0.03 (n.s.)** — the
  representation does not move the headline rating; mode comparisons measure
  the music, not the format.
- Largest dimension shift: structure **+0.15** (ABC's repeat/section syntax is
  more legible than a flat note list).
- Secondary analysis: toolkit code-gen pieces judged as ABC vs native ABC
  pieces are within-model equivalent (−0.000 ± 0.125 across 10 models).

Conversion uses the vendored `xml2abc_177/xml2abc.py` (Willem Vree, LGPL) with
flags `-m 2 -d 8 -n 999999` plus stripping of `transpose=` voice directives.

## 2. Judge-prompt pilot: composer persona + topline placement

`judge_prompt_pilot.py` compares the old judge system prompt against the
composer-persona prompt (no enumerated dimension list) with the top-line score
asked LAST, on a seeded piece set with test–retest repeats.

Findings (`prompt_pilot_results.json`):
- Retest noise: composer persona 0.230 vs 0.298 (old prompt) at equal
  discrimination — the persona prompt was adopted.
- Topline-asked-last paired effect on other dimensions: structure −0.107,
  emotion −0.060; all else null.

Full judging checkpoints and run logs are not committed; they live in the
corpus backup (`~/llm-music-backup/pod/format_ab/`).
