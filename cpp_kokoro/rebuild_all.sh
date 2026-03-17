#!/bin/bash
set -e

# Get script directory (works regardless of where the script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# Configuration
ONNX_VERSION="1.24.2"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"
ONNX_DIR="$SCRIPT_DIR/onnx"
ONNX_TGZ="$ONNX_DIR/onnxruntime-linux-x64-gpu-${ONNX_VERSION}.tgz"
ONNX_EXTRACTED="$ONNX_DIR/onnxruntime-linux-x64-gpu-${ONNX_VERSION}"
ONNX_CMAKE_DIR="$ONNX_EXTRACTED/lib/cmake/onnxruntime"

MODEL_URL="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/onnx/model.onnx"
VOICE_URL="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices/af.bin"

echo "=== Step 0: Install system dependencies ==="
sudo apt-get install -y build-essential cmake wget pkg-config \
    portaudio19-dev espeak-ng

echo "=== Step 1: Clean up existing directories ==="
rm -rf "$BUILD_DIR"
rm -rf "$ONNX_DIR"

echo "=== Step 2: Download and extract ONNX Runtime ==="
mkdir -p "$ONNX_DIR"
wget -O "$ONNX_TGZ" "$ONNX_URL"
tar -xzf "$ONNX_TGZ" -C "$ONNX_DIR"
rm -f "$ONNX_TGZ"

echo "=== Step 3: Fix ONNX Runtime paths ==="
mkdir -p "$ONNX_EXTRACTED/lib64"
ln -sf "$ONNX_EXTRACTED/lib/libonnxruntime.so.${ONNX_VERSION}" \
       "$ONNX_EXTRACTED/lib64/libonnxruntime.so.${ONNX_VERSION}"
cp "$ONNX_CMAKE_DIR/onnxruntimeConfig.cmake" \
   "$ONNX_CMAKE_DIR/ONNXRuntimeConfig.cmake"

echo "=== Step 4: Download model and voice ==="
mkdir -p "$BUILD_DIR/voices"
if [ ! -f "$BUILD_DIR/kokoro-v1.0.onnx" ]; then
    wget -O "$BUILD_DIR/kokoro-v1.0.onnx" "$MODEL_URL"
fi
if [ ! -f "$BUILD_DIR/voices/af.bin" ]; then
    wget -P "$BUILD_DIR/voices/" "$VOICE_URL"
fi

echo "=== Step 5: Configure and build ==="
cd "$BUILD_DIR"
cmake ..
cmake --build .

echo ""
echo "=== Build complete ==="
echo "Run:  cd $BUILD_DIR && ./kokoro_tts"
