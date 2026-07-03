#!/bin/bash
# Set up Music2Emo on a CUDA GPU pod (tested: RTX A4000, driver 550 / CUDA 12.4).
# Reproduces the audio-emotion leg. Run from anywhere; expects this repo checked out
# so it can copy in music2emo_patch.py. Assumes python3.11 + git available.
#
#   bash setup_music2emo_gpu.sh /path/to/llm-music-expression
#
# After this, run scripts/music2emo_extract.py from ~/Music2Emotion (see its header).
set -e
REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

echo "### GPU / driver ###"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || { echo "no GPU"; exit 1; }

echo "### venv + torch (cu124 matches a CUDA 12.x driver; pick the wheel for your driver) ###"
python3.11 -m venv ~/m2e-venv
source ~/m2e-venv/bin/activate
pip install -q --upgrade pip wheel numpy
pip install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

echo "### Music2Emotion repo + reqs (minus its pinned torch) + librosa ###"
[ -d ~/Music2Emotion ] || git clone --depth 1 https://github.com/AMAAI-Lab/Music2Emotion.git ~/Music2Emotion
cd ~/Music2Emotion
grep -vE "^(torch==|torchaudio==)" requirements.txt > /tmp/req.txt
pip install -q -r /tmp/req.txt
pip install -q librosa

echo "### patch music2emo.py + verify predict() exposes the full key set ###"
cp "$REPO/scripts/music2emo_patch.py" ~/Music2Emotion/
python music2emo_patch.py
python - <<'PY'
import torch
_o=torch.load; torch.load=lambda *a,**k:_o(*a,**{**k,"weights_only":False})
from music2emo import Music2emo
out=Music2emo().predict("inference/input/test.mp3")
need=("valence","arousal","predicted_moods","mood_probs","key","chords","mert_embedding")
missing=[k for k in need if k not in out]
print("predict keys:", list(out.keys()))
print("=== M2E SETUP OK ===" if not missing else f"=== SETUP INCOMPLETE, missing {missing} ===")
PY
