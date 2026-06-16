#!/usr/bin/env bash
# Download a General MIDI SoundFont for FluidSynth audio rendering.
# Also install FluidSynth itself if missing (macOS: brew, Debian/Ubuntu: apt).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF_DIR="$ROOT/soundfonts"
SF_PATH="$SF_DIR/FluidR3_GM.sf2"
SF_URL="https://archive.org/download/fluidr3-gm-gs/FluidR3_GM.sf2"

if ! command -v fluidsynth >/dev/null 2>&1; then
  echo "FluidSynth not found."
  if command -v brew >/dev/null 2>&1; then
    echo "Installing via Homebrew…"; brew install fluid-synth
  elif command -v apt-get >/dev/null 2>&1; then
    echo "Installing via apt…"; sudo apt-get update && sudo apt-get install -y fluidsynth
  else
    echo "Please install FluidSynth manually, then re-run."; exit 1
  fi
fi

mkdir -p "$SF_DIR"
if [ -f "$SF_PATH" ]; then
  echo "SoundFont already present: $SF_PATH"
else
  echo "Downloading SoundFont to $SF_PATH …"
  curl -L --fail -o "$SF_PATH" "$SF_URL"
fi
echo "Done. Audio rendering is ready."
