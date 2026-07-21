#!/bin/bash
# One-shot bootstrap + launcher for the composer studio on a RunPod CPU pod
# (mirrors scripts/runpod_bootstrap.sh conventions: idempotent, onstart-able).
#
# Pod shape: on-demand (NOT spot) CPU pod, 2 vCPU / 4 GB, ~20 GB volume,
# port 8321 exposed as an HTTP port. RunPod's proxy then serves the studio at
#   https://<pod-id>-8321.proxy.runpod.net
# with TLS included — no domain or reverse proxy needed. Set this script as
# the pod's "Docker command"/onstart, or paste it into the web terminal.
#
# Needs ~/llm-music-expression/.env (scp it like the experiment pods):
#   ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY
#   STUDIO_PASSWORD, STUDIO_SECRET  (see .env.example)
# Optional: STUDIO_BACKUP_REPO — a private git remote (https URL with token,
# e.g. https://<PAT>@github.com/<you>/studio-backup.git); sessions are pushed
# there hourly. Pod volumes survive stop/start but NOT pod termination, so
# set it.
set -e

# --- authorize the working key (account keys inject only at pod CREATION) ---
mkdir -p ~/.ssh && chmod 700 ~/.ssh
KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIP+hcTUqjMMR46Cb485L3mHsuRSfX7/f5W42isVEnPFa nathaniel_kang@brown.edu'
grep -qF "$KEY" ~/.ssh/authorized_keys 2>/dev/null || echo "$KEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# --- system deps: renderers for the full audio path --------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git curl abcmidi fluidsynth lame

# --- python >= 3.11 (RunPod's 20.04 base images ship 3.8) ---------------------
PY=python3
if ! $PY -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    apt-get install -y -qq software-properties-common
    add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
    apt-get update -qq
    apt-get install -y -qq python3.11 python3.11-venv
    PY=python3.11
fi

# --- repo + venv + package ----------------------------------------------------
cd ~
[ -d llm-music-expression ] || git clone https://github.com/nateykang/llm-music-expression.git
cd llm-music-expression
git pull --ff-only || true
[ -d .venv ] || $PY -m venv .venv  # never recreate: 3.8's venv would clobber 3.11's
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[studio]"

# --- soundfont (idempotent; skipped if already present) -----------------------
ls soundfonts/*.sf2 >/dev/null 2>&1 || bash scripts/setup_soundfont.sh

# Config can come from a scp'd .env OR from pod environment variables set in
# the RunPod console (Edit Pod -> Environment Variables). Console env vars are
# the durable option: they survive the container wipes that config edits cause,
# making the pod fully self-healing when this script is the start command.
if [ ! -f .env ] && [ -z "${STUDIO_PASSWORD:-}" ]; then
    echo "!!! no ~/llm-music-expression/.env and no STUDIO_PASSWORD env var —" >&2
    echo "    scp a .env over, or set env vars in the RunPod console." >&2
    exit 1
fi

# --- backup loop (hourly, if STUDIO_BACKUP_REPO is set) -----------------------
# .env takes precedence over the console env var: console edits wipe the
# container, so a leaked/rotated token must be swappable via a pod-local .env
# line without touching pod config.
BACKUP_REPO="$(grep -E '^STUDIO_BACKUP_REPO=' .env 2>/dev/null | cut -d= -f2-)"
BACKUP_REPO="${BACKUP_REPO:-${STUDIO_BACKUP_REPO:-}}"
# Console env-var fields love to smuggle in trailing newlines; a URL with one
# makes git fail. Secrets never contain whitespace, so strip it all.
BACKUP_REPO="$(printf '%s' "$BACKUP_REPO" | tr -d '[:space:]')"
if [ -n "$BACKUP_REPO" ]; then
    (
        set +e  # the loop must outlive transient failures (set -e is inherited)
        # Never prompt for credentials (a prompt in this headless loop hangs it
        # forever — observed as backups silently stopping), and bound every
        # network call so a wedged connection can't freeze the loop either.
        export GIT_TERMINAL_PROMPT=0
        while true; do
            if [ -d studio_data ]; then
                # A clone that failed half-way (or an rsync before it) leaves a
                # non-git ~/studio_backup that blocks every later clone — reset it.
                if [ ! -d ~/studio_backup/.git ]; then
                    rm -rf ~/studio_backup
                    timeout 300 git clone -q "$BACKUP_REPO" ~/studio_backup \
                        || echo "backup: clone failed — check STUDIO_BACKUP_REPO" >&2
                fi
                if [ -d ~/studio_backup/.git ]; then
                    rsync -a studio_data/ ~/studio_backup/studio_data/
                    (
                        cd ~/studio_backup || exit
                        git add -A
                        git -c user.email=studio@pod -c user.name=studio \
                            commit -qm "backup $(date -u +%Y-%m-%dT%H:%M)" 2>/dev/null
                        timeout 300 git push -q || echo "backup: push failed" >&2
                    )
                fi
            fi
            sleep 1h
        done
    ) &
    # NEVER print the URL itself: it embeds the access token, and this log has
    # already leaked one token via exactly that mistake.
    echo "backup loop running (hourly -> $(printf '%s' "$BACKUP_REPO" | sed -E 's#//[^@/]+@#//[token]@#'))"
else
    echo "WARNING: STUDIO_BACKUP_REPO unset — sessions die with the pod." >&2
fi

# --- run the studio under a restart loop (no systemd in pod containers) -------
echo "=== studio starting on 0.0.0.0:8321 ==="
echo "URL: https://${RUNPOD_POD_ID:-<pod-id>}-8321.proxy.runpod.net"
while true; do
    .venv/bin/llm-music-studio --host 0.0.0.0 --port 8321 --proxy-headers || true
    echo "studio exited — restarting in 5s" >&2
    sleep 5
done
