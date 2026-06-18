# Representing & Generating Music with LLMs — Literature Notes

*A grounding review for this project: prompting frontier LLMs (Opus 4.8, GPT-5.5, …)
to compose music to probe their biases, comparing a **music21 code-generation** mode
against an **ABC-notation** mode. Sources are peer-reviewed / arXiv primary papers,
adversarially fact-checked (22 of 25 candidate claims survived 3-vote verification).*

---

## TL;DR

- **ABC notation is the field standard for text-based LLM music** — but as *raw text
  rendered by an ABC-native engine*, never piped through a code library's importer.
  (That pipe is exactly what broke our ABC mode.)
- **Multi-voice misalignment is a documented, citable failure mode** — not a quirk of
  our setup — but it's *practical, not fundamental*, and purpose-built systems fix it
  with synchronized/interleaved ABC.
- **No one has published a head-to-head of code/DSL-generation vs. ABC for zero-shot
  frontier LLMs.** Our code-vs-ABC axis is a genuine gap to fill.
- **Almost every empirical number in the literature is from 2024 models** (GPT-4,
  GPT-3.5, Llama2). Treat them as priors and known failure modes, *not* predictions
  about Opus 4.8 / GPT-5.5. Re-measuring on frontier models is itself the contribution.

---

## 1. How the literature represents music for LLMs

| Representation | Who uses it | Tradeoffs |
|---|---|---|
| **ABC notation** | ChatMusician, MuPT, NotaGen, ComposerX | Compact character-level text; plain text tokenizer; **~288 tokens/song (~38% of MIDI)**; weak at polyphony/simultaneity |
| **Tokenized MIDI** (REMI, event/AMT) | Music Transformer, MuseNet, Midi-LLM | Captures timing/velocity richly; long sequences; dedicated systems adapt the embedding layer rather than serialize MIDI as text |
| **MusicXML / \*\*kern / MEI** | mostly MIR & analysis | Rich notation; verbose; rarely used for LLM generation |
| **Code / DSL** (music21, Sonic Pi) | sara-fish; some agent/tool-use work | Reusable abstractions + a render/execute loop; **niche, under-studied** |

**Key finding (high confidence):** "Most text-to-symbolic-music tasks currently process
an ABC notation, as this encoding is already in a textual format" (ACM Computing Surveys
survey). ABC is chosen because it is "a pure text tokenizer without any external
multi-modal neural structures" (ChatMusician), and MuPT argues "LLMs are inherently more
compatible with ABC Notation, which aligns more closely with their design and strengths."

**Token efficiency (high confidence):** ChatMusician measured ABC at **288.21 avg
tokens/song ≈ 38% of MIDI-based representations.**

> **Implication for us:** routing ABC through music21's secondary ABC *importer* was the
> wrong call — the field treats ABC as raw text → ABC-native renderer (e.g. abcjs).
> music21 is the right tool for the *code* path (the model writes a music21 program),
> not for parsing ABC.

---

## 2. Key systems & papers

- **MuPT** — *A Generative Symbolic Music Pretrained Transformer* (arXiv:2404.06393).
  Argues ABC > MIDI for LM pretraining; introduces **SMT-ABC (Synchronized Multi-Track
  ABC)** to "address the challenges associated with misaligned measures from different
  tracks." *Context: training a dedicated model, ABC-vs-MIDI — not ABC-vs-code, not
  zero-shot.*
- **ChatMusician** (arXiv:2402.16153) — LLaMA2 + continual ABC pretraining; pure text
  tokenizer. Self-reports surpassing 2024 GPT-4 on conditioned composition. *(Benchmark
  superiority is author-reported; one college-level-theory superiority claim was
  refuted in verification — treat cautiously.)*
- **NotaGen** (arXiv:2502.18008) — interleaved ABC: "different voices of the same bar are
  rearranged into a single line… ensures alignment of duration and musical content
  across voices."
- **ComposerX** (arXiv:2404.18081) — multi-agent, GPT-4-turbo, **training-free** ABC.
  Explicitly documents inter-voice alignment failures, out-of-range notes, and
  notation-reality gaps.
- **Midi-LLM** (arXiv:2511.03942) — expands Llama-3.2-1B's embedding layer for MIDI
  tokens rather than serializing MIDI as text; picks AMT tokenization because it "does
  not require beat-synchronized data that is a prerequisite for REMI and ABC-based
  approaches." *Supports choosing ABC or code over raw tokenized MIDI for prompting.*

---

## 3. Probing biases / "self-expression" via music generation

This subfield is **thin** in the verified literature.

- **Closest peer-reviewed anchor:** Shin & Kaneko, *Large Language Models' Internal
  Perception of Symbolic Music* (arXiv:2507.12808). Generates MIDI from genre/style
  prompts with no explicit musical training; finds LLMs "can infer rudimentary musical
  structures and temporal relationships" but are limited "due to a lack of explicit
  musical context."
- **ISMIR 2024** (arXiv:2407.21531): "current LLMs exhibit poor performance in song-level
  multi-step music reasoning… advanced musical capability is not intrinsically obtained
  by LLMs." *(Tested GPT-4 / Llama2 — predates frontier models.)*
- **The sara-fish "LLM musical self-expression" line was NOT found by any verified
  source.** It appears to be a project/blog, not citable peer-reviewed work — cite it as
  inspiration for the *setup*, not as literature.

---

## 4. ABC vs. code vs. MIDI for frontier models — what's actually known

- **Zero-shot ABC validity is imperfect (high confidence):** on 500 "respond in ABC"
  prompts, format-correctness was **GPT-4 94.6%, GPT-3.5 65.4%**, vs. **fine-tuned
  ChatMusician 99.6%** (ChatMusician paper). → A validity-check + render/repair loop is
  standard practice. *(A claim that "only GPT-4 exceeds 50% renderable ABC" was refuted.)*
- **Multi-voice is the central failure (high confidence)** — but **"fundamental
  limitation" was refuted (0–3 votes).** Frame it as a *practical, mitigable* failure
  mode (SMT-ABC / interleaved ABC), not an absolute one.
- **No verified source compares music21-code vs. ABC head-to-head for zero-shot frontier
  LLMs.** Open gap → our experimental variable.

---

## Caveats & refuted claims (read before citing)

- **Time-sensitivity dominates.** ChatMusician format rates, ISMIR reasoning failures,
  GPT-4 harmony weakness — all 2024 models. Motivating priors, not predictions for
  Opus 4.8 / GPT-5.5.
- **Stable (not time-sensitive):** ABC token efficiency, character-level text
  compatibility, polyphony serialization challenges, SMT-ABC / interleaving.
- **Refuted in verification:** (1) ChatMusician beats GPT-4 on college-level
  MusicTheoryBench (0–3); (2) "only GPT-4 >50% renderable ABC" (1–2); (3) ABC's linear
  text nature is a *fundamental* polyphony limitation (0–3).

## Open questions this project can address

1. Do **Opus 4.8 / GPT-5.5** still show multi-voice ABC misalignment and sub-100% ABC
   validity, or has scale closed the 2024 gaps?
2. Does generating **music21 code** (interpreter enforces valid structure, renders
   deterministically) reduce syntactic-validity and alignment failures vs. free-text ABC?
3. Head-to-head **code-gen vs. ABC** for frontier-model composition — apparently unstudied.

## Primary sources

- MuPT — https://arxiv.org/abs/2404.06393
- ChatMusician — https://arxiv.org/abs/2402.16153
- NotaGen — https://arxiv.org/abs/2502.18008
- ComposerX — https://arxiv.org/abs/2404.18081
- Midi-LLM — https://arxiv.org/abs/2511.03942
- LLMs' Internal Perception of Symbolic Music — https://arxiv.org/abs/2507.12808
- ISMIR 2024 (LLM music reasoning) — https://arxiv.org/abs/2407.21531
- ACM Computing Surveys (symbolic music + NLP) — https://dl.acm.org/doi/10.1145/3714457

*Method: 5 search angles → 17 primary sources fetched → 66 claims extracted → 25
verified with 3-vote adversarial checking (need 2/3 to refute) → 22 confirmed.*
