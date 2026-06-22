# Observations — building intuition on LLM music output

*Running notes from building this project. The point is to separate **what the
models actually do** from **what our pipeline made it look like they do** — those
got confused over and over, and untangling them is most of the intuition.*

Confidence tags: **[proven]** = measured/verified, **[impression]** = from a few
examples, treat as a hypothesis to test by ear.

---

## The one lesson that kept repeating: the renderer can lie

Almost every "the model is bad at X" turned out to be "our pipeline mangled good
output." Before trusting any judgment about a model, rule out the rendering path.
The confounds we hit, each one a trap:

| Symptom we saw | Real cause | Verdict |
|---|---|---|
| gpt-5.5 string quartet = one unlabeled piano staff | **music21's ABC importer collapses multi-voice ABC** — dumped all notes into one voice, left the other 3 as all-rests | Model wrote a real 4-voice quartet (421 notes); renderer hid it. **[proven]** |
| Code-gen audio missing the piano's bass/left hand | **music21's MIDI export drops a grand-staff's 2nd part** while its MusicXML keeps it | Pipeline bug. Fixed by deriving MIDI *from* the MusicXML. **[proven]** |
| Quartet plays as all piano | **abcjs defaults every voice to piano** without `%%MIDI program` directives (the `name="Violin"` field is notation-only, no playback meaning) | Renderer default. Fixed by injecting GM programs. **[proven]** |
| opus postmodern: sheet + audio don't match the description | opus wrote `[V1]` instead of `[V:V1]` (missing colon) → abcjs reads `[` as a chord and scrambles voices | A transcription typo, not a musical choice. **[proven]** |
| "The bass isn't playing" | Bass written very low (E1 ≈ 41 Hz); **laptop speakers can't reproduce <~150 Hz** | Bass *was* there — 9× the low-freq energy in the mix. Use headphones. **[proven]** |

**Takeaway:** when something sounds/looks wrong, the first question is "is this the
model or the renderer?" — not "this model is bad at music." We only trust a
judgment after the rendering path is verified faithful.

---

## Genuine model failures (distinct from the above)

These are real — the pipeline rendered exactly what the model wrote:

- **gpt-4.1 cannot write a runnable fugue in code.** Hallucinated non-existent
  music21 methods (`getRealizationOfDegree`, `getElementsByNumber`) and crashed
  across 6 attempts → shows "—". **[proven]**
- **gpt-4.1's ABC fugue is hollow.** It "succeeded" syntactically, but 2 of the 3
  voices are *entirely rests* — a fugue in name only, one real line over two empty
  staves. **[proven]**
- **gpt-4.1 wrote malformed ABC** ("Night Reflections", stab-voicing): octave marks
  in the wrong place (`B2,` instead of `B,2`), a non-standard `%% score {}`
  directive → unrenderable. Now surfaced as an honest error + raw ABC. **[proven]**
- **Even Opus occasionally fumbles ABC syntax** (the `[V1]` colon slip) — 1 of 44
  pieces. Frontier models still emit invalid ABC at low rates. **[proven]**

**Loud vs. quiet failure** is the throughline: code fails *loudly* (interpreter
crashes → retry loop catches it), ABC fails *quietly* (passes a coarse syntax
gate, then renders blank or hollow). ABC's "robustness" can **mask** incompetence.

---

## What each representation elicits (the interesting part)

The format isn't neutral — it changes *what the model bothers to specify*:

- **Instrumentation: code forces it, ABC makes it optional.** In music21 you must
  write `instrument.Violin()` — there's no default-piano escape hatch. In ABC you
  can just label a staff `name="Violin"` and skip the playback instrument — and
  **most models take that out**, hence "all piano." Code forces commitment; ABC
  permits omission, and the models reveal their laziness through which they do.
  **[proven mechanism; "models are lazy here" is impression]**
- **Bias vs. reliability (the toolkit tension).** Code-gen *needs* a music21 cheat
  sheet in the prompt (valid instruments + GM numbers, duration tables, a
  grand-staff recipe) or models hallucinate APIs and crash. But that cheat sheet
  *is* a set of seeded defaults — it nudges instrument/rhythm/texture choices,
  contaminating the "self-expression" we're trying to measure. **ABC needs almost
  none of that scaffolding → cleaner probe, but less reliable.** So code-vs-ABC is
  partly a bias-vs-reliability trade.
- **SMT-ABC (synchronized/interleaved).** Writing each bar's voices together (and
  asking for `%%MIDI program` per voice) is the literature's fix for multi-voice
  drift. Models *do* follow the format when prompted. Whether it produces
  *better-aligned* music than plain ABC on frontier models is the open question to
  judge by ear. **[infrastructure proven; outcome TBD]**

---

## Per-model impressions (tentative — few examples each)

All **[impression]** — hypotheses to confirm by listening across more cells:

- **Opus 4.8** — strongest. Real multi-voice quartets, nailed the fugue in code.
  Rare ABC syntax slip. Leans introspective / minor-key.
- **GPT-5.5** — solid multi-voice (genuine 4-voice quartet), coherent structure.
- **Sonnet 4.6** — not deeply examined yet; add notes.
- **GPT-4.1** — the weakest / best stress-test: code-fugue crash, hollow ABC fugue,
  malformed ABC. Where the cracks show.
- **Cross-model:** strong pull toward **D minor / introspective / "self-reflection"**
  affect in free-form, across models. Possibly the single most interesting bias
  signal — worth probing directly.

---

## Grounding from the literature (see literature-review.md)

- ABC is the field standard for *text-LLM* music (token-efficient, text-native).
- Multi-voice misalignment is a **documented** failure mode — *practical, not
  fundamental*; SMT-ABC / interleaved ABC is the known fix.
- Zero-shot ABC validity is imperfect (2024: GPT-4 94.6%, GPT-3.5 65.4%).
- General LLMs encode only "rudimentary" musical structure (2024 models).
- **No published code-vs-ABC head-to-head for frontier models** → genuine gap this
  project can fill.
- **Caveat:** all literature numbers are 2024 models (GPT-4/Llama2), not Opus 4.8 /
  GPT-5.5 — priors and failure modes to instrument for, not predictions.

---

## Open questions to judge by ear (not yet quantified — deliberately)

1. **Does SMT-ABC actually keep voices aligned** vs plain ABC, on these models?
   (string-quartet, fugue — toggle ABC ↔ SMT-ABC and listen for drift.)
2. Does SMT-ABC's bar-by-bar scaffolding **coax inner voices** out of a weak model
   (gpt-4.1's hollow fugue)?
3. **Free-form across all three methods, same model** — what does the model reach
   for when only the representation changes? (purest bias probe)
4. Is the **D-minor / introspective** pull real and cross-model, or an artifact of
   the self-expression framing?

> Quantify *after* forming hunches by ear — metrics should test hypotheses, not
> manufacture them. (A descriptor/heatmap pass is ready to build when that time
> comes.)

---

## Process notes (not about the music, but bit us)

- **Batches die on laptop sleep.** `caffeinate -i` blocks idle sleep but **not**
  lid-close. Incremental writes mean a killed run loses nothing; `scripts/
  resume_batch.py` finishes the missing cells.
- Generation needs the machine awake + online + the session alive — **can't run
  offline.**
