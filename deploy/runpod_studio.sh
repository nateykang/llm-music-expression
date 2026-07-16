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
# there every 6 h. Pod volumes survive stop/start but NOT pod termination, so
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
apt-get install -y -qq git abcmidi fluidsynth lame

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

if [ ! -f .env ]; then
    echo "!!! ~/llm-music-expression/.env missing — scp it over, then re-run." >&2
    exit 1
fi

# --- backup loop (every 6 h, if STUDIO_BACKUP_REPO is set in .env) ------------
BACKUP_REPO=$(grep -E '^STUDIO_BACKUP_REPO=' .env | cut -d= -f2-)
if [ -n "$BACKUP_REPO" ]; then
    (
        while true; do
            if [ -d studio_data ]; then
                [ -d ~/studio_backup ] || git clone -q "$BACKUP_REPO" ~/studio_backup || true
                if [ -d ~/studio_backup/.git ]; then
                    rsync -a studio_data/ ~/studio_backup/studio_data/
                    cd ~/studio_backup
                    git add -A
                    git -c user.email=studio@pod -c user.name=studio \
                        commit -qm "backup $(date -u +%Y-%m-%dT%H:%M)" 2>/dev/null || true
                    git push -q || true
                    cd ~/llm-music-expression
                fi
            fi
            sleep 6h
        done
    ) &
    echo "backup loop running (every 6h -> $BACKUP_REPO)"
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
