#!/usr/bin/env bash
# Part 2 extension on a pod: generate the new arms' express-yourself pieces
# (ABC + codegen, 20 samples each, same settings as the v3 corpus), push them,
# then fill every missing judging cell: the old panel on the new pieces and the
# new arms on every express-yourself piece. Every stage is resumable; re-run
# the script after a hiccup and it picks up where it stopped.
#
# Prereqs on the pod: scripts/runpod_setup.sh done; .env with ANTHROPIC_API_KEY,
# OPENROUTER_API_KEY (+ OPENAI_API_KEY if any arm is native OpenAI) and
# GITHUB_TOKEN; the Mac's Part 2 files copied in:
#   part2_run/analysis/part2_selfpref_ckpt.json   (required — else the old
#   part2_run/analysis/part2_selfpref_raw.json     panel x old pieces would be
#   part2_run/analysis/part2_selfpref.csv          re-judged from scratch)
#   part2_run/reps_cache.json                      (optional, saves ~3 min)
#
# Env knobs:
#   NEW_ARMS     default: the six fable-5.1 / opus-5.5 / gpt-6-astra arms
#   SKIP_JUDGES  comma list left off the panel (e.g. opus-4.8,opus-4.8-thinking)
#   STAGE        probe | generate | judge | all (default all)
#   WORKERS      generation + realtime judge concurrency (default 8)
#
# Launch detached:  setsid bash scripts/runpod_part2_extend.sh </dev/null >/dev/null 2>&1 &
# Progress:         tail -f part2_extend.log
set -uo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"
. .venv/bin/activate
set -a; [ -f .env ] && . ./.env; set +a
export LLM_MUSIC_ROOT="$PWD"

NEW_ARMS="${NEW_ARMS:-fable-5.1,fable-5.1-thinking,opus-5.5,opus-5.5-thinking,gpt-6-astra,gpt-6-astra-thinking}"
STAGE="${STAGE:-all}"
W="${WORKERS:-8}"
STATE=part2_run/extend_batches.txt   # the generated batch dirs, one per line
LOG=part2_extend.log
REPO_SLUG="${REPO_SLUG:-nateykang/llm-music-expression}"
BRANCH="${BRANCH:-claude/fable-opus-gpt6-pipeline-fcj5h4}"
exec >>"$LOG" 2>&1
say() { echo "=== $(date -u +%FT%TZ) $*"; }

push() {  # commit + push given paths to the working branch
  [ -z "${GITHUB_TOKEN:-}" ] && { say "GITHUB_TOKEN unset — not pushing ($1)"; return 0; }
  git add -A "${@:2}"
  git -c user.email="pod@runpod" -c user.name="runpod" commit -q -m "$1" || return 0
  for d in 2 4 8 16; do
    git push -q "https://x-access-token:${GITHUB_TOKEN}@github.com/${REPO_SLUG}.git" "HEAD:${BRANCH}" && return 0
    sleep "$d"
  done
  say "push failed ($1)"
}

for k in ANTHROPIC_API_KEY OPENROUTER_API_KEY; do
  [ -n "${!k:-}" ] || { say "missing $k in .env"; exit 1; }
done

if [[ "$STAGE" == probe || "$STAGE" == all ]]; then
  say "probe: one live judge call per new arm"
  python part2_run/probe_arms.py ${NEW_ARMS//,/ } || { say "probe FAILED — fix registry ids before generating"; exit 1; }
fi

gen_mode() {  # $1 = abc|codegen; resumes an unfinished batch of this mode if recorded
  local mode=$1 dir
  dir=$(grep "|$mode$" "$STATE" 2>/dev/null | tail -1 | cut -d'|' -f1)
  if [ -n "$dir" ]; then
    say "generate $mode: resuming $dir"
    llm-music batch --models "$NEW_ARMS" --prompts express-yourself --mode "$mode" \
      --samples 20 --workers "$W" --no-audio --resume "docs/data/$dir"
  else
    local before after
    before=$(ls docs/data)
    say "generate $mode: $NEW_ARMS x express-yourself x 20"
    # record the new dir as soon as it appears, so a crash mid-run can resume it
    ( while sleep 5; do
        after=$(comm -13 <(echo "$before") <(ls docs/data) | grep __models_ | head -1)
        [ -n "$after" ] && { echo "$after|$mode" >> "$STATE"; break; }
      done ) &
    local watcher=$!
    llm-music batch --models "$NEW_ARMS" --prompts express-yourself --mode "$mode" \
      --samples 20 --workers "$W" --no-audio
    sleep 6; kill "$watcher" 2>/dev/null
  fi
}

if [[ "$STAGE" == generate || "$STAGE" == all ]]; then
  gen_mode abc
  gen_mode codegen
  python - <<'EOF'
import json
for line in open("part2_run/extend_batches.txt"):
    d = line.strip().split("|")[0]
    ps = json.load(open(f"docs/data/{d}/data.json"))["pieces"]
    ok = sum(p.get("ok", False) for p in ps)
    print(f"{d}: {ok}/{len(ps)} ok")
EOF
  push "Part 2 extension: generated express-yourself batches for $NEW_ARMS" docs/data "$STATE"
fi

if [[ "$STAGE" == judge || "$STAGE" == all ]]; then
  [ -f part2_run/analysis/part2_selfpref_ckpt.json ] || {
    say "part2_selfpref_ckpt.json missing — copy it from the Mac first"; exit 1; }
  export PART2_EXTRA_BATCHES=$(cut -d'|' -f1 "$STATE" | paste -sd, -)
  export PART2_SKIP_JUDGES="${SKIP_JUDGES:-}"
  export PART2_TAG=ext1
  say "judge: extra batches $PART2_EXTRA_BATCHES; skipping judges: ${PART2_SKIP_JUDGES:-none}"
  cp part2_run/analysis/part2_selfpref_ckpt.json "part2_run/analysis/part2_selfpref_ckpt.pre_ext1.json" 2>/dev/null || true
  # Batch legs run one after another: each read-modify-writes the shared checkpoint.
  say "leg 1/3: Anthropic Message Batches"; python part2_run/batch_anthropic.py
  say "leg 2/3: OpenRouter batches";        python part2_run/batch_openrouter.py
  # Realtime sweep over the full panel: only cells still missing are called
  # (arms without a batch route, plus anything a batch leg failed), then the
  # raw JSON + CSV are rebuilt from the complete checkpoint.
  say "leg 3/3: realtime fill + aggregate";  WORKERS="$W" python part2_run/run.py
  # The CSV always goes up; raw JSON only while under GitHub's 100 MB file cap.
  # The checkpoint (with reasoning traces) stays on the pod: scp it back.
  files=(part2_run/analysis/part2_selfpref.csv)
  raw=part2_run/analysis/part2_selfpref_raw.json
  [ -f "$raw" ] && [ "$(stat -c %s "$raw")" -lt 95000000 ] && files+=("$raw")
  push "Part 2 extension: judging results (ext1)" "${files[@]}"
  say "checkpoint left on pod: part2_run/analysis/part2_selfpref_ckpt.json (scp it back)"
fi
say "done (stage=$STAGE)"
