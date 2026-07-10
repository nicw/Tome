#!/usr/bin/env bash
# Setup for granite shadow transcription (spec: docs/superpowers/specs/2026-07-09-granite-shadow-transcription-design.md)
set -euo pipefail

MODEL_DIR="$HOME/Library/Application Support/Tome/Granite"
REPO="https://huggingface.co/ibm-granite/granite-speech-4.1-2b-GGUF/resolve/main"
MODEL="granite-speech-4.1-2b-Q8_0.gguf"     # keep in sync with ShadowConfig.modelFilename
MMPROJ="mmproj-model-f16.gguf"              # keep in sync with ShadowConfig.mmprojFilename
SERVER="${LLAMA_SERVER:-/opt/homebrew/bin/llama-server}"
PORT="${GRANITE_PORT:-8873}"

if [[ ! -x "$SERVER" ]]; then
  echo "llama-server not found at $SERVER — run: brew install llama.cpp" >&2
  exit 1
fi
# b9045+ required for granite-speech mtmd support
# --version prints either "bNNNNN" (older builds) or "version: NNNNN (hash)"
# (current homebrew llama.cpp) — match either form. `|| true` on each grep
# guards against `set -e` killing the script when a pattern doesn't match.
VERSION_OUTPUT=$("$SERVER" --version 2>&1 || true)
BUILD=$(printf '%s' "$VERSION_OUTPUT" | grep -oE 'b[0-9]+' | head -1 | tr -d 'b' || true)
if [[ -z "$BUILD" ]]; then
  BUILD=$(printf '%s' "$VERSION_OUTPUT" | grep -oE 'version: [0-9]+' | head -1 | grep -oE '[0-9]+' || true)
fi
if [[ -z "$BUILD" || "$BUILD" -lt 9045 ]]; then
  echo "llama.cpp build b${BUILD:-unknown} < b9045 — run: brew upgrade llama.cpp" >&2
  exit 1
fi

mkdir -p "$MODEL_DIR"
for f in "$MODEL" "$MMPROJ"; do
  echo "Downloading $f (resumable)…"
  curl -L -C - --fail -o "$MODEL_DIR/$f" "$REPO/$f"
done
ls -lh "$MODEL_DIR"

cat <<EOF

Setup complete. Enable shadow mode with:
  defaults write com.dloomis.tome graniteShadowEnabled -bool YES
(Disable: defaults write com.dloomis.tome graniteShadowEnabled -bool NO)

Proof-of-life (Ctrl-C to stop the server when done):
  "$SERVER" -m "$MODEL_DIR/$MODEL" --mmproj "$MODEL_DIR/$MMPROJ" --host 127.0.0.1 --port $PORT
EOF
