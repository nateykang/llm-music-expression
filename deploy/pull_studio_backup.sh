#!/bin/bash
# Mac-side studio backup: pull studio_data off the pod over ssh, commit + push
# it to the private backup repo with the Mac's own git credentials (the pod
# holds no token). Loops every INTERVAL seconds; pass --once for a single pass.
#
# Usage: deploy/pull_studio_backup.sh <ssh-host> <ssh-port> <label> [--once]
set -u
HOST="$1"; PORT="$2"; LABEL="$3"; ONCE="${4:-}"
REPO="${STUDIO_BACKUP_DIR:-$HOME/llm-music-studio-backup}"
INTERVAL="${INTERVAL:-600}"
export GIT_TERMINAL_PROMPT=0

while true; do
    mkdir -p "$REPO/$LABEL"
    if rsync -a -e "ssh -p $PORT -i $HOME/.ssh/id_ed25519 -o ConnectTimeout=20 -o BatchMode=yes" \
            "root@$HOST:llm-music-expression/studio_data/" "$REPO/$LABEL/studio_data/"; then
        (
            cd "$REPO" || exit
            git add -A "$LABEL"
            git commit -qm "backup $LABEL $(date -u +%Y-%m-%dT%H:%M)" 2>/dev/null \
                && { git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 push -q && echo "$(date) pushed" || echo "$(date) push failed" >&2; }
        )
    else
        echo "$(date) rsync failed (pod down or ssh port changed?)" >&2
    fi
    [ "$ONCE" = "--once" ] && break
    sleep "$INTERVAL"
done
