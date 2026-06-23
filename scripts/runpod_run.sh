#!/usr/bin/env bash
# Run a sampling batch on the pod and push the results back to GitHub on a fresh
# branch (conflict-free — your laptop just fetches it). Because the pod never
# sleeps and isn't tied to any session, this runs to completion unattended.
#
# Configure via env vars (all optional except the keys + token):
#   MODELS    comma-separated model ids   (default: the 7 fast frontier models)
#   PROMPTS   comma-separated prompts      (default: free-form)
#   MODE      codegen | abc | smt-abc      (default: abc)
#   SAMPLES   repeats per cell             (default: 30)
#   GITHUB_TOKEN   PAT with repo write     (required to push results)
#   REPO_SLUG owner/repo                   (default: nateykang/llm-music-expression)
#
# Usage:  bash scripts/runpod_run.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

MODELS="${MODELS:-gpt-5.5,gpt-4.1,gemini-2.5-pro,grok-4.3,deepseek-v4-pro,qwen3-max,llama-4-maverick}"
PROMPTS="${PROMPTS:-free-form}"
MODE="${MODE:-abc}"
SAMPLES="${SAMPLES:-30}"
WORKERS="${WORKERS:-8}"
REPO_SLUG="${REPO_SLUG:-nateykang/llm-music-expression}"

# Sampling corpora feed the metrics dashboard, not the site player, so skip audio
# by default (keeps the repo/Pages lean). Set AUDIO=1 to bake MP3s anyway.
AUDIO_FLAG="--no-audio"; [ -n "${AUDIO:-}" ] && AUDIO_FLAG=""
echo "=== generating: $MODELS x $PROMPTS x $SAMPLES samples [$MODE], $WORKERS workers ${AUDIO_FLAG} ==="
llm-music batch --models "$MODELS" --prompts "$PROMPTS" --mode "$MODE" \
  --samples "$SAMPLES" --workers "$WORKERS" $AUDIO_FLAG

echo "=== pushing results ==="
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN not set — results are on disk under docs/data but NOT pushed."
  echo "Set GITHUB_TOKEN and re-run the push, or copy docs/data off the pod manually."
  exit 0
fi
BRANCH="runpod-data-$(date +%Y%m%d_%H%M%S)"
git add -A docs/data
git -c user.email="pod@runpod" -c user.name="runpod" \
    commit -q -m "Sampling run: $MODELS x $PROMPTS x $SAMPLES [$MODE]"
git push -q "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_SLUG}.git" "HEAD:${BRANCH}"
echo ""
echo "=== done. Pushed to branch: ${BRANCH} ==="
echo "On your laptop:  git fetch origin ${BRANCH} && git checkout ${BRANCH} -- docs/data"
