#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu/Debian box (a RunPod CPU pod is plenty — no GPU
# needed; the value is an always-on machine that never sleeps and isn't tied to
# your laptop session). Installs system deps, clones the repo, builds the venv,
# and downloads a SoundFont so generated pieces get audio.
#
# Usage on the pod:
#   export REPO_URL=https://github.com/nateykang/llm-music-expression.git   # optional
#   bash runpod_setup.sh        # (or: curl -sSL <raw-url>/runpod_setup.sh | bash)
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/nateykang/llm-music-expression.git}"
WORKDIR="${WORKDIR:-$HOME}"

echo "=== [1/4] system dependencies ==="
export DEBIAN_FRONTEND=noninteractive
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -qq
# python + git, plus the audio toolchain (abc2midi for ABC->MIDI, fluidsynth+lame
# for MP3). abc2midi is also needed later for metric analysis.
$SUDO apt-get install -y -qq \
  git python3 python3-venv python3-pip \
  abcmidi fluidsynth lame curl

echo "=== [2/4] clone repo ==="
cd "$WORKDIR"
[ -d llm-music-expression ] || git clone "$REPO_URL"
cd llm-music-expression

echo "=== [3/4] python env ==="
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e .

echo "=== [4/4] soundfont ==="
bash scripts/setup_soundfont.sh || echo "(soundfont setup failed — pieces will generate without audio, metrics still work)"

echo ""
echo "=== Setup complete. ==="
echo "Next: export your API keys, then run scripts/runpod_run.sh"
echo "  export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  OPENROUTER_API_KEY=..."
echo "  export GITHUB_TOKEN=...        # a PAT with repo write, to push results back"
