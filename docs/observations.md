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
| gpt-5.5 string quartet = one unlabeled piano staff | **music21's ABC importer collapses multi-voice ABC** — dumped all notes into one voice, the other 3 became all-rests | Model wrote a real 4-voice quartet (421 notes); renderer hid it. **[proven]** |
| Code-gen audio missing the piano's bass/left hand | **music21's MIDI export drops a grand-staff's 2nd part** while its MusicXML keeps it | Pipeline bug; fixed by deriving MIDI *from* the MusicXML. **[proven]** |
| Quartet plays as all piano | **abcjs/abc2midi default every voice to piano** without `%%MIDI program` (the `name="Violin"` field is notation-only) | Renderer default; fixed by injecting GM programs from voice names. **[proven]** |
| gpt-5.5 stab plays as piano, not voices | Model labeled `name=Soprano` **unquoted**; our instrument regex only matched `name="Soprano"` | Our bug; model wrote proper SATB (even `clef=treble-8` for tenor). **[proven]** |
| opus postmodern: sheet + audio don't match the description | opus wrote `[V1]` instead of `[V:V1]` (missing colon) → abcjs reads `[` as a chord and scrambles voices | A transcription typo, not a musical choice. **[proven]** |
| gpt-5.5 fugue sheet music is *tiny* | Model wrote each voice's 41 bars on **one ABC line**; abcjs scaled the giant system down to ~0.2× | Renderer default (honors source line breaks); fixed with abcjs `wrap`. **[proven]** |
| gpt-5.5 code-gen stab = **no audio** in Chrome | FluidSynth's **Ogg** output has broken length metadata (reported 254s for 67s) → Chrome won't play it | Pipeline bug; fixed by switching all audio to **MP3**. **[proven]** |
| "The bass isn't playing" | Bass written very low (E1 ≈ 41 Hz); **laptop speakers can't reproduce <~150 Hz** | Bass *was* there — 9× the low-freq energy in the mix. Use headphones. **[proven]** |

**Takeaway:** when something sounds/looks wrong, first ask "is this the model or the
renderer?" — not "this model is bad at music." We only trust a judgment after the
rendering path is verified faithful.

**Status:** as of now the rendering path *is* trustworthy — we moved to the standard
ABC toolchain (see "How we render," below), fixed every confound above, and made all
three modes pre-bake MP3. So new "this sounds wrong" observations are now far more
likely to be the **model**, which is the footing we wanted for intuition.

---

## Genuine model FAILURES (the pipeline rendered exactly what was written)

- **gpt-4.1 cannot write a runnable fugue in code.** Hallucinated non-existent
  music21 methods (`getRealizationOfDegree`, `getElementsByNumber`) and crashed
  across 6 attempts → shows "—". **[proven]**
- **gpt-4.1's ABC fugue is hollow.** Syntactically valid, but 2 of the 3 voices are
  *entirely rests* — a fugue in name only, one real line over two empty staves.
  **[proven]**
- **gpt-4.1 wrote malformed ABC** ("Night Reflections", stab): octave marks in the
  wrong place (`B2,` vs `B,2`), a non-standard `%% score {}` directive →
  unrenderable. Now surfaced as an honest error + raw ABC. **[proven]**
- **Even Opus occasionally fumbles ABC syntax** (the `[V1]` colon slip) — 1 of 44.
  Frontier models still emit invalid ABC at low rates. **[proven]**

**Loud vs. quiet failure** is the throughline: code fails *loudly* (interpreter
crashes → retry loop catches it), ABC fails *quietly* (passes a coarse syntax gate,
then renders blank or hollow). ABC's "robustness" can **mask** incompetence.

---

## Genuine model SUCCESSES (real musical competence, verified)

Balances the failures above — these are things the models did *well*, confirmed
against the source once the renderer was trustworthy:

- **gpt-5.5 writes a real fugue.** ABC three-part fugue has a proper fugal
  exposition: subject in voice 1, the **answer entering a few bars later** in voice
  2, bass entering later still. Staggered entries = an actual fugue, the opposite of
  gpt-4.1's hollow one. **[proven]**
- **gpt-5.5 deploys real harmonic devices** (Western harmony, "Lantern in the
  Machine"): starts in **D harmonic minor** (correct raised leading tone, `^c`),
  then modulates to **D major** for the final third — a deliberate **parallel-major
  / Picardy** brightening. Notably it *distributed the harmonic weight*: the
  accompaniment's tonic triad audibly flips minor→major (obvious), while the melody
  brightens gradually (subtle) — sounds like the parts "switch at different times"
  but they're notated together. That's voice-leading craft, not a bug. Even the
  voice names (cantus / inner light / ground) are idiomatic. **[proven]**
- **gpt-5.5 knows vocal notation.** Stab piece used SATB with `clef=treble-8` (the
  octave-down tenor clef) — a detail you only use if you understand choral scoring.
  **[proven]**

The pattern: gpt-5.5 in particular shows genuine theory literacy (harmonic minor,
modulation, fugal form, choral clefs) — well beyond "rudimentary," and a useful
counter to the 2024-era literature claim that LLMs lack real musical structure.

---

## What each representation elicits (the interesting part)

The format isn't neutral — it changes *what the model bothers to specify*:

- **Instrumentation: code forces it, ABC makes it optional.** In music21 you must
  write `instrument.Violin()` — no default-piano escape hatch. In ABC you can label
  a staff and skip the playback instrument, and **most models do**, hence "all
  piano." Code forces commitment; ABC permits omission. **[proven mechanism]**
- **ABC syntax is fiddly and models slip.** Missing colons (`[V1]`), unquoted names
  (`name=Soprano`), misplaced octave marks (`B2,`), whole voices on one line. None
  are *musical* errors — they're transcription slips — but they break strict tools.
  Code's failures are louder; ABC's are subtler. **[proven]**
- **Bias vs. reliability (the toolkit tension).** Code-gen *needs* a music21 cheat
  sheet in the prompt (instruments + GM numbers, duration tables, grand-staff
  recipe) or models hallucinate APIs and crash — but that cheat sheet *seeds
  defaults* that contaminate "self-expression." **ABC needs almost none of that →
  cleaner probe, less reliable.** Code-vs-ABC is partly a bias-vs-reliability trade.
- **SMT-ABC (synchronized/interleaved).** Writing each bar's voices together (+
  `%%MIDI program` per voice) is the literature's fix for multi-voice drift. Models
  follow the format when asked. Whether it produces *better-aligned* music than
  plain ABC on frontier models is still the open question to judge by ear.
  **[infrastructure proven; outcome TBD]**

---

## Per-model impressions (tentative — confirm across more cells)

- **Opus 4.8** — strongest overall. Real multi-voice quartets, nailed the fugue in
  code. Rare ABC syntax slip (`[V1]`). Leans introspective / minor-key. **[impression]**
- **GPT-5.5** — clear theory literacy: real fugue (staggered entries), parallel-major
  modulation with harmonic-minor leading tones, choral clefs. ABC habits: writes
  *unquoted* voice names and whole voices on one line. **[impression → leaning proven on the specific pieces]**
- **Sonnet 4.6** — not deeply examined; add notes as you listen.
- **GPT-4.1** — weakest / best stress-test: code-fugue crash, hollow ABC fugue,
  malformed ABC. Where the cracks show. **[proven on those pieces]**
- **Cross-model:** strong pull toward **D minor / introspective / "self-reflection"**
  in free-form. Possibly the single most interesting bias signal — probe directly.
  **[impression]**

---

## Grounding from the literature (see literature-review.md)

- ABC is the field standard for *text-LLM* music (token-efficient, text-native).
- Multi-voice misalignment is a **documented** failure mode — *practical, not
  fundamental*; SMT-ABC / interleaved ABC is the known fix.
- Zero-shot ABC validity is imperfect (2024: GPT-4 94.6%, GPT-3.5 65.4%) — and we
  saw it ourselves (even Opus slipped once; gpt-4.1 several times).
- General LLMs encode only "rudimentary" musical structure (2024 models) — but
  gpt-5.5's harmonic-minor modulation and real fugue suggest frontier models have
  moved past that.
- **No published code-vs-ABC head-to-head for frontier models** → genuine gap this
  project can fill.
- **Caveat:** literature numbers are 2024 models (GPT-4/Llama2), not Opus 4.8 /
  GPT-5.5 — priors and failure modes to instrument for, not predictions.

---

## How we render now (so we can trust what we hear)

We converged on the **standard ABC toolchain** (as used by EasyABC / ChatMusician) —
*not* music21, whose weak ABC importer caused the worst confounds:

| Mode | Notation | Audio |
|---|---|---|
| Code-gen | Verovio (MusicXML) | music21 → MIDI → FluidSynth → **MP3** |
| ABC | **abcjs** (raw ABC) | **abc2midi** → FluidSynth → **MP3** |
| SMT-ABC | **abcjs** (raw ABC) | **abc2midi** → FluidSynth → **MP3** |

`abcjs` for notation, `abc2midi` for audio — the canonical split. All three modes
pre-bake MP3 (correct duration, plays in every browser incl. Safari/iOS). Instrument
programs come from the model's `%%MIDI program` or, when omitted, are injected from
voice names; bare `[V1]` markers are normalized to `[V:V1]` first.

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
5. Is **gpt-5.5's harmonic sophistication** (modulation, fugal form) consistent, or
   did we just catch good examples?

> Quantify *after* forming hunches by ear — metrics should test hypotheses, not
> manufacture them. (Feature-measurement is the next phase.)

---

## Process notes (not about the music, but bit us)

- **Batches die on laptop sleep.** `caffeinate -i` blocks idle sleep but **not**
  lid-close. Incremental writes mean a killed run loses nothing; `scripts/
  resume_batch.py` finishes the missing cells.
- Generation needs the machine awake + online + the session alive — **can't run
  offline.**
- Many UI bugs were the HTML `[hidden]` attribute losing to `display:flex/grid`
  (stale compare grid, controls not hiding) — fixed globally with
  `[hidden]{display:none!important}`. Audio needed explicit cross-engine stop/pause
  (native `<audio>` + Web Audio synth) on switch and on play.
