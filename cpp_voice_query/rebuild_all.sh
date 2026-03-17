#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check sibling projects exist
if [ ! -d "../cpp_whisper/whisper.cpp" ]; then
    echo "ERROR: ../cpp_whisper/whisper.cpp not found."
    echo "Run ../cpp_whisper/rebuild_all.sh first."
    exit 1
fi

if [ ! -d "../cpp_kokoro/onnx" ]; then
    echo "ERROR: ../cpp_kokoro/onnx not found."
    echo "Run ../cpp_kokoro/rebuild_all.sh first."
    exit 1
fi

# Install system dependencies (skip if no sudo)
echo "=== Installing system dependencies ==="
sudo apt-get install -y libcurl4-openssl-dev portaudio19-dev espeak-ng 2>/dev/null || \
    echo "Note: could not install deps (no sudo). Ensure libcurl4-openssl-dev, portaudio19-dev, espeak-ng are installed."

# Download nlohmann/json header if not present
mkdir -p include
if [ ! -f "include/nlohmann/json.hpp" ]; then
    echo "=== Downloading nlohmann/json.hpp ==="
    mkdir -p include/nlohmann
    curl -L -o include/nlohmann/json.hpp \
        "https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp"
else
    echo "=== nlohmann/json.hpp already downloaded ==="
fi

# Build
echo "=== Building ==="
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)

echo ""
echo "=== Build complete ==="
echo "Run: ./build/voice_query"
echo "  Options:"
echo "    --whisper-model PATH   (default: ../cpp_whisper/models/ggml-base.bin)"
echo "    --language LANG        (default: pt)  [en, pt]"
echo "    --llm-url URL          (default: http://localhost:8001)"
echo "    --no-tts               Disable spoken responses"
echo "    --text-only            Just transcribe, skip LLM"
