# Composer Studio

A password-gated chat web app where a collaborator (a composer, no programming
background) converses with a model that composes through this repo's existing
pipeline: music21 sandbox / ABC → MusicXML + MIDI → FluidSynth MP3, engraved in
the browser with Verovio (codegen) or abcjs (ABC). The composer picks the
writing method per message (a code/ABC toggle above the chat box — the model is
only offered the matching render tool, so the choice is really theirs). Every
session is logged to disk as research data.

## Run locally

```bash
pip install -e ".[studio]"          # or: uv sync --extra studio
export STUDIO_PASSWORD=changeme     # or put it in .env (dotenv is loaded)
python -m llm_music.studio          # http://127.0.0.1:8321
```

`ANTHROPIC_API_KEY` must be set (same `.env` as the batch pipeline). Audio
baking needs FluidSynth + a SoundFont + lame (`scripts/setup_soundfont.sh`);
without them scores still engrave, only MP3s are skipped. ABC audio also wants
`abc2midi` (`brew install abcmidi` / `apt install abcmidi`).

## Environment variables

| var | default | meaning |
|---|---|---|
| `STUDIO_PASSWORD` | *(required)* | the shared password |
| `ANTHROPIC_API_KEY` | *(required)* | agent turns run on the Anthropic API |
| `STUDIO_DATA_DIR` | `./studio_data` | sessions, event logs, rendered pieces |
| `STUDIO_SECRET` | random per boot | pin to keep login cookies valid across restarts |
| `STUDIO_MODELS` | all anthropic registry models | comma list of friendly ids to offer |
| `STUDIO_DEFAULT_MODEL` | `opus-4.8` | preselected model for new sessions (non-thinking, so turns stay snappy; pick fable-5 in the UI when quality is worth minutes of silent thinking) |
| `STUDIO_NOTIFY_URL` | *(off)* | webhook POSTed on session create / resume-after-idle |

Models come from [`registry.py`](src/llm_music/models/registry.py); only
Anthropic entries qualify (the agent loop uses tool use on that API). Each
session starts on the model picked at creation, and the dropdown above the chat
box switches it mid-conversation (applies from the next message and sticks).
Prior turns' thinking blocks are stripped from the replayed context — their
signatures are model-specific — but stay in events.jsonl.

## Deploy on a small VPS

A $5/month CPU box (Hetzner CX22, DigitalOcean basic) is plenty — the heavy
lifting is API tokens, not compute.

```bash
# as root, once
adduser --disabled-password studio
apt install -y python3.12-venv fluidsynth lame abcmidi caddy

# as studio
git clone <this repo> ~/llm-music-expression && cd ~/llm-music-expression
python3 -m venv .venv && .venv/bin/pip install -e ".[studio]"
bash scripts/setup_soundfont.sh
cp .env.example .env   # add ANTHROPIC_API_KEY, STUDIO_PASSWORD, STUDIO_SECRET

# as root
cp deploy/studio.service /etc/systemd/system/ && systemctl enable --now studio
cp deploy/Caddyfile /etc/caddy/Caddyfile   # edit hostname first
systemctl reload caddy
```

Point a DNS A record (e.g. `studio.yourdomain.com`) at the box; Caddy gets the
TLS certificate automatically. Link that URL from the main site's nav — the
"tab" is just a link.

## Spend guardrails

The password protects your API budget, so belt and suspenders:

- Set a monthly spend cap on the key's workspace in the Anthropic console.
- Turns are capped at 8 model calls (`MAX_TURN_STEPS` in
  [`studio/config.py`](src/llm_music/studio/config.py)).
- Login is rate-limited (5 failures / 15 min per IP).

## The research data

Everything lands in `STUDIO_DATA_DIR/sessions/<id>/`:

- `events.jsonl` — composer messages, assistant text, **full thinking traces**,
  every tool call with complete source, render errors, token usage. This is
  the artifact to study; the UI shows only the user/assistant/piece subset.
- `messages.json` — raw Anthropic transcript (resumes the chat across visits).
- `pieces/vN/` — source (`source.py` or `piece.abc`), MusicXML/MIDI, MP3,
  meta.json with the model's own revision note + quick analysis.

Back it up nightly, e.g. in the `studio` user's crontab:

```cron
15 3 * * * rsync -a ~/llm-music-expression/studio_data/ ~/studio_backup/ && tar czf ~/studio_backup.tgz ~/studio_backup
```

(or rsync to your laptop / commit to a private repo — anything off-box.)

## Roadmap (deliberately not built yet)

- Click-a-measure annotations: Verovio SVG carries `xml:id`s that map back to
  MusicXML elements, so a selection + comment can be injected into the agent's
  context as structured feedback.
- Playback cursor synced to the score (Verovio MIDI-time → element mapping).
- WebMIDI sketch input.

Watch how he actually works first; his sessions will tell you which of these
to build.
