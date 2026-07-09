#!/bin/bash
# Run the two judging experiments back-to-back, unattended, on a RunPod CPU box.
# Both are resumable (content-keyed checkpoints), so re-running continues where
# it left off — safe to launch again after a pod hiccup.
#
# Launch detached so it survives SSH logout:
#   cd ~/llm-music-expression
#   setsid bash scripts/run_experiments.sh </dev/null >/dev/null 2>&1 &
#
# Progress + a marker file per stage land in the repo root:
#   run_experiments.log, .bach_done, .relabel_done
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
. .venv/bin/activate 2>/dev/null || true
W="${WORKERS:-6}"
LOG="run_experiments.log"

{
  echo "=== experiments start $(date -u) (workers=$W) ==="

  echo "=== bach start $(date -u) ==="
  python scripts/judge_bach.py --workers "$W"
  echo "=== bach end $(date -u) rc=$? ==="
  date -u > .bach_done

  echo "=== relabel start $(date -u) ==="
  python scripts/judge_relabel.py --workers "$W"
  echo "=== relabel end $(date -u) rc=$? ==="
  date -u > .relabel_done

  echo "=== experiments all done $(date -u) ==="
} >> "$LOG" 2>&1
