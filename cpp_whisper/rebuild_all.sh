#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Install system dependencies (skip if sudo not available)
echo "=== Installing system dependencies ==="
sudo apt-get install -y portaudio19-dev cmake build-essential 2>/dev/null || \
    echo "Note: could not install deps (no sudo). Ensure portaudio19-dev and cmake are installed."

# Clone whisper.cpp if not present
if [ ! -d "whisper.cpp" ]; then
    echo "=== Cloning whisper.cpp ==="
    git clone https://github.com/ggerganov/whisper.cpp.git
else
    echo "=== whisper.cpp already cloned ==="
fi

# Download Whisper model
mkdir -p models
BASE_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

echo ""
echo "=== Select Whisper model ==="
echo ""
echo "  1) tiny    (~75 MB)  - Fastest, lowest accuracy"
echo "  2) base    (~148 MB) - Good balance for quick tasks"
echo "  3) small   (~466 MB) - Good accuracy, moderate speed"
echo "  4) medium  (~1.5 GB) - Very good accuracy"
echo "  5) large-v3 (~3.1 GB) - Best accuracy, slowest"
echo ""
echo "Larger models are more accurate, especially with accents or noisy audio."
echo "For fine-tuning to your voice, train in Python and convert to GGUF."
echo ""
read -p "Choose model [1-5] (default: 2): " choice

case "${choice:-2}" in
    1) MODEL_NAME="ggml-tiny.bin" ;;
    2) MODEL_NAME="ggml-base.bin" ;;
    3) MODEL_NAME="ggml-small.bin" ;;
    4) MODEL_NAME="ggml-medium.bin" ;;
    5) MODEL_NAME="ggml-large-v3.bin" ;;
    *) echo "Invalid choice, using base."; MODEL_NAME="ggml-small.bin" ;;
esac

MODEL="models/$MODEL_NAME"
if [ ! -f "$MODEL" ]; then
    echo "=== Downloading $MODEL_NAME ==="
    curl -L -o "$MODEL" "$BASE_URL/$MODEL_NAME"
else
    echo "=== $MODEL_NAME already downloaded ==="
fi

# Build
echo "=== Building ==="
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Run: ./build/whisper_stt models/$MODEL_NAME"
