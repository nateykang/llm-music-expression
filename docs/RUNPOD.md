# Running generation unattended (RunPod / any cloud box)

Long sampling runs shouldn't depend on your laptop staying awake. A small **CPU**
pod (no GPU — the models run on the providers' servers; the pod just orchestrates
API calls + light rendering) is an always-on machine that never sleeps and isn't
tied to any editor session. It generates the corpus and pushes results back to a
GitHub branch; your laptop just fetches them.

## 1. Spin up a pod
Any Ubuntu/Debian box works. On RunPod: a cheap CPU pod (or a spot GPU pod left on
CPU) with an Ubuntu image and a web terminal / SSH.

## 2. Set secrets (in the pod's env or a `.env`)
```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
export GITHUB_TOKEN=...      # GitHub PAT with repo write — lets the pod push results
```

## 3. Set up + run
```bash
# fetch the setup script and run it (clones repo, builds venv, installs deps + soundfont)
curl -sSL https://raw.githubusercontent.com/nateykang/llm-music-expression/main/scripts/runpod_setup.sh | bash
cd ~/llm-music-expression

# generate (defaults: 7 fast models x free-form x 30 samples, ABC). Override via env:
MODELS=opus-4.8,opus-4.8-thinking,gpt-5.5 SAMPLES=30 bash scripts/runpod_run.sh
```
The run finishes unattended (the pod never sleeps), then pushes to a fresh branch
like `runpod-data-20260622_193000` and prints the exact fetch command.

## 4. Pull results on your laptop
```bash
git fetch origin runpod-data-<stamp>
git checkout runpod-data-<stamp> -- docs/data      # grab just the generated data
```
Then locally: `llm-music analyze docs/data/<new-batch>` + `llm-music report` to fold
it into the dashboard. (Metrics aggregate across all batches by model, so partial /
multiple pod runs combine automatically.)

## Notes
- **No GPU needed.** Generation is API calls + CPU rendering. A GPU pod only helps if
  you later self-host an open model (e.g. a ChatMusician/XMIDI emotion classifier).
- Each pod run pushes to its **own branch**, so runs never conflict with each other
  or with your laptop's `main`.
- If `GITHUB_TOKEN` is unset the run still completes — results sit in `docs/data` on
  the pod for you to copy off manually.
