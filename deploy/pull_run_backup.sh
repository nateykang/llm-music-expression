#!/bin/bash
# Mac-side backup of a pod run: rsync the run directory to this repo, then push
# the corpus metadata, batch handles, logs and per-judge verdict shards to the
# private backup repo with the Mac's git credentials (the pod holds no token).
#
# Usage: deploy/pull_run_backup.sh <ssh-host> <ssh-port> <run-dir> [--once]
set -u
HOST="$1"; PORT="$2"; RUN="$3"; ONCE="${4:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="${STUDIO_BACKUP_DIR:-$HOME/llm-music-studio-backup}"
INTERVAL="${INTERVAL:-600}"
SSH="ssh -p $PORT -i $HOME/.ssh/id_ed25519 -o ConnectTimeout=20 -o BatchMode=yes"
export GIT_TERMINAL_PROMPT=0

while true; do
    if rsync -a --exclude '*.tmp' --exclude data/ --exclude __pycache__/ -e "$SSH" \
            "root@$HOST:llm-music-expression/$RUN/" "$ROOT/$RUN/"; then
        "$ROOT/.venv/bin/python" "$ROOT/scripts/backup_run_shards.py" "$ROOT" "$RUN" "$REPO/$RUN" \
            && (
                cd "$REPO" || exit
                git add -A "$RUN"
                git commit -qm "backup $RUN $(date -u +%Y-%m-%dT%H:%M)" 2>/dev/null || exit 0
                git pull -q --rebase && git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=120 push -q \
                    && echo "$(date) pushed $(git rev-parse --short HEAD)" || echo "$(date) PUSH FAILED" >&2
            )
    else
        echo "$(date) RSYNC FAILED (pod down or ssh port changed?)" >&2
    fi
    [ "$ONCE" = "--once" ] && break
    sleep "$INTERVAL"
done
