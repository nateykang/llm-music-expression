# LLM Musical Self-Expression

*How do LLMs express themselves musically?* A rebuild of
[sara-fish/llm-musical-self-expression](https://github.com/sara-fish/llm-musical-self-expression)
that improves three things:

1. **Live engraving with [Verovio](https://www.verovio.org/)** — scores are stored as
   MusicXML and engraved in the browser (MuseScore-tier quality, no Lilypond/MuseScore
   install, interactive). Verovio can also export PDF/SVG client-side.
2. **Two generation modes** — sandboxed `music21` **code-gen** (faithful default) and an
   **ABC-direct** mode with no code execution (safe, reproducible, good for comparison).
3. **A one-line model registry** — adding the newest model is a single entry in
   [`registry.py`](src/llm_music/models/registry.py).

It stays **batch-and-bake / fully static**, like the original: a CLI generates every
model × prompt pair once into `docs/data/<timestamp>/`, and the static site just serves
those files. Hitting *play* streams a pre-rendered audio file — no runtime model calls.

## How it works

```
prompt ──▶ model ──▶ ┌ codegen: JSON{code,…} ─▶ sandbox runs music21 ─┐
                     └ abc:     JSON{abc,…}  ─▶ music21.converter ─────┘
                                                         │
                          music21 Score ─▶ MIDI + MusicXML
                                            │        │
                          FluidSynth ◀──────┘        └──▶ Verovio (browser)
                              │
                            audio.mp3  ──▶ docs/data/<ts>/ + data.json ──▶ static site
```

## Setup

```bash
# 1. Install (pick one)
uv sync                       # if you use uv
pip install -e ".[dev]"       # plain pip

# 2. API key
cp .env.example .env          # then add ANTHROPIC_API_KEY

# 3. (optional) audio rendering
bash scripts/setup_soundfont.sh   # installs FluidSynth + a SoundFont
```

Without FluidSynth/SoundFont the pipeline still produces scores and the site still
engraves them — only the pre-baked audio is skipped.

## Usage

```bash
llm-music models                                    # list registered models

# one piece
llm-music run --model sonnet-4.6 --prompt free-form --mode abc

# a matrix (the original's batch-and-bake)
llm-music batch --models opus-4.8,sonnet-4.6 \
                --prompts free-form,fugue,string-quartet --mode codegen
```

Output lands in `docs/data/<timestamp>__models_N_prompts_M/`. The batch folder and
its `data.json` are written incrementally — one piece at a time — so an interrupted
run still leaves a valid, viewable partial batch.

### Prompts

Prompts mirror [sara-fish's set](https://github.com/sara-fish/llm-musical-self-expression)
verbatim. The shared frame lives in [`prompts/prompt.md`](prompts/prompt.md); the
eleven form/structure instructions are one row each in
[`prompts/form_instructions.csv`](prompts/form_instructions.csv):

```
free-form · piano-piece · harp-piece · western-harmony · fugue · stab-voicing
string-quartet · orchestral · melody-accomp · modern · postmodern
```

Codegen mode also injects a music21 cheat-sheet ([`prompts/toolkit.md`](prompts/toolkit.md))
into the prompt to cut failed attempts. Add a prompt by appending a CSV row — no code
changes.

## View the site

```bash
python scripts/serve.py 8000   # then open http://localhost:8000
```

Pick batch / prompt / model; Verovio engraves the score live and the audio plays.
`docs/` is GitHub Pages-ready as-is.

> All modes pre-bake audio to **MP3** and play it with a native `<audio>` tag.
> The split follows the standard ABC toolchain (as in EasyABC/ChatMusician):
> **notation** comes from Verovio (code-gen MusicXML) or **abcjs** (ABC), and
> **audio** from FluidSynth — fed by music21's MIDI for code-gen and by
> **`abc2midi`** (the reference ABC→MIDI converter) for ABC. MP3 (FluidSynth → WAV
> → `lame`) is used over FluidSynth's direct Ogg, whose broken length header makes
> browsers misreport the duration and breaks Chrome playback; MP3 also plays on
> Safari/iOS. Requires `abc2midi` and `lame` (`brew install abcmidi lame`); without
> them scores still render, only the pre-baked audio is skipped (ABC then falls
> back to the abcjs in-browser synth).

## Adding a model

Append one line to [`MODEL_REGISTRY`](src/llm_music/models/registry.py):

```python
"haiku-4.5": ("anthropic", "claude-haiku-4-5-20251001"),
```

**Anthropic** and **OpenAI** adapters ship in the box. The OpenAI one
([`openai.py`](src/llm_music/models/openai.py)) uses the Responses API, so it covers
both standard chat models (`gpt-4.1`, `gpt-4o`) and reasoning models (`o3`, GPT-5-class)
— set `OPENAI_API_KEY` in `.env` and adjust the registry ids to whatever your org grants.

A further **provider** (OpenRouter, Ollama, …) = a new adapter module implementing the
`LLMClient` protocol ([`base.py`](src/llm_music/models/base.py)) plus a branch in
`_build_client`. Nothing else changes.

## Safety note

Code-gen mode executes model-written Python. It runs in an isolated subprocess
([`sandbox.py`](src/llm_music/sandbox.py)) with CPU/memory limits and a wall-clock
timeout. For untrusted/large runs, run inside a container as a second layer. ABC mode
executes no code.

## Tests

```bash
pytest
```
