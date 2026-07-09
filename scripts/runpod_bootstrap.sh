#!/bin/bash
# One-shot pod bootstrap so a fresh RunPod CPU box is turnkey for the judging
# runs — no manual SSH-key or dependency steps per pod.
#
# Set this as the pod template's "Docker command" / onstart, OR paste it once
# into the RunPod web terminal. It is idempotent: safe to re-run.
#
# It authorizes the key RunPod injection may have missed (account keys are only
# injected at pod CREATION, not on restart — the recurring gotcha), installs the
# system + Python deps, and clones the repo. After it finishes, connect over SSH
# and launch the experiments (see scripts/judge_bach.py / judge_relabel.py).
set -e

# --- authorize the working key (idempotent) ---------------------------------
mkdir -p ~/.ssh && chmod 700 ~/.ssh
KEY='ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIP+hcTUqjMMR46Cb485L3mHsuRSfX7/f5W42isVEnPFa nathaniel_kang@brown.edu'
grep -qF "$KEY" ~/.ssh/authorized_keys 2>/dev/null || echo "$KEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# --- system deps: git + abc2midi (the judge renders ABC instrument headers) --
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git abcmidi

# --- repo + venv + package -------------------------------------------------
cd ~
[ -d llm-music-expression ] || git clone https://github.com/nateykang/llm-music-expression.git
cd llm-music-expression
git pull --ff-only || true
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e ".[dev]"

echo "=== bootstrap complete ==="
echo "still need ~/llm-music-expression/.env (API keys) — scp it from your laptop:"
echo "  scp -P <port> -i ~/.ssh/id_ed25519 .env root@<ip>:~/llm-music-expression/.env"
echo "then: cd ~/llm-music-expression && . .venv/bin/activate && python scripts/judge_bach.py --limit 20"
