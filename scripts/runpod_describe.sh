#!/usr/bin/env bash
# Overnight description-arms experiment on the pod: generate post-hoc
# descriptions for the full corpus (free-form + sparse-toolkit batches, author
# model describes its own piece), then run the blind valence ratings and the
# per-piece content contrast, then push data + analysis back to GitHub on a
# fresh branch. Every stage is checkpointed, so re-running after any failure
# resumes where it left off.
#
#   GITHUB_TOKEN   PAT with repo write     (required to push results)
#   REPO_SLUG      owner/repo              (default: nateykang/llm-music-expression)
#   WORKERS        rating/contrast threads (default: 6)
#
# Usage:  bash scripts/runpod_describe.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q vaderSentiment

WORKERS="${WORKERS:-6}"
REPO_SLUG="${REPO_SLUG:-nateykang/llm-music-expression}"

BATCHES=(
  20260622_164100__models_11_prompts_1
  20260622_195241__models_7_prompts_1
  20260627_045417__models_1_prompts_1
  20260623_105811__models_11_prompts_1
  20260720_014910__models_13_prompts_1
  20260720_025945__models_13_prompts_1
  20260720_151138__models_13_prompts_1
)

echo "=== phase 0: post-hoc descriptions (${#BATCHES[@]} batches in parallel) ==="
pids=()
for b in "${BATCHES[@]}"; do
  python -m llm_music.cli redescribe "docs/data/$b" \
    > "redescribe_$b.log" 2>&1 &
  pids+=($!)
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
tail -n 1 redescribe_*.log
[ "$fail" = 1 ] && echo "warning: some redescribe pieces failed (see logs); continuing"

echo "=== phase 1: blind valence ratings ==="
python scripts/compare_description_arms.py --workers "$WORKERS" \
  2>&1 | grep -v "INFO httpx" | tail -20

echo "=== phase 2: content contrast ==="
python scripts/contrast_descriptions.py --workers "$WORKERS" \
  2>&1 | grep -v "INFO httpx" | tail -20

echo "=== pushing results ==="
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "GITHUB_TOKEN not set — results are on disk (docs/data, docs/analysis) but NOT pushed."
  exit 0
fi
BRANCH="runpod-desc-$(date +%Y%m%d_%H%M%S)"
git add -A docs/data docs/analysis
git -c user.email="pod@runpod" -c user.name="runpod" \
    commit -q -m "Description-arms overnight run: redescribe + ratings + contrast"
git push -q "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_SLUG}.git" "HEAD:${BRANCH}"
echo ""
echo "=== done. Pushed to branch: ${BRANCH} ==="
echo "On your laptop:  git fetch origin ${BRANCH} && git checkout ${BRANCH} -- docs/data docs/analysis"
