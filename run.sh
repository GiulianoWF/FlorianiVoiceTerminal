#!/bin/bash
# Starts llama-server + proxy inspector
# llama-server on :8001, proxy on :9001, web UI on :9002
#
# Usage:
#   ./run.sh           Full mode  (Qwen3-Coder-Next, MoE offload)
#   ./run.sh --light   Light mode (Qwen3-32B, fully on GPU)
#   ./run.sh -l        Same as --light

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="/home/giulianowf/Documents/llama.cpp/build/bin/llama-server"
PROXY="$SCRIPT_DIR/proxy.py"

# Context size presets
CONTEXT_32K=32768     # ~408MB KV  — lightweight, fastest prompt processing
CONTEXT_64K=65536     # ~816MB KV  — minimum for Claude Code
CONTEXT_128K=131072   # ~1632MB KV — current sweet spot (recommended)
CONTEXT_256K=262144   # ~3264MB KV — full model capacity, may need fewer experts on GPU

# Parse arguments
MODE="full"
for arg in "$@"; do
    case "$arg" in
        --light|-l) MODE="light" ;;
        *) echo "Unknown option: $arg"; echo "Usage: $0 [--light|-l]"; exit 1 ;;
    esac
done

# Configure per mode
if [ "$MODE" = "light" ]; then
    MODEL="$SCRIPT_DIR/models/qwen3-32b/Qwen_Qwen3-32B-Q4_K_M.gguf"
    ALIAS="qwen3-32b"
    CONTEXT=$CONTEXT_32K
    EXTRA_ARGS=()
    SAMPLING_ARGS=(--temp 0.6 --top-k 20 --top-p 0.95 --min-p 0)
else
    MODEL="$SCRIPT_DIR/Qwen3-Coder-Next-GGUF/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"
    ALIAS="qwen3-coder-next"
    CONTEXT=$CONTEXT_32K
    EXTRA_ARGS=(-ot "blk\.3[0-9]\.ffn_.*_exps.=CPU,blk\.4[0-7]\.ffn_.*_exps.=CPU")
    SAMPLING_ARGS=(--temp 1.0 --top-p 0.95 --min-p 0.01 --top-k 40)
fi

cleanup() {
    echo ""
    echo "Shutting down..."
    kill $LLAMA_PID $PROXY_PID 2>/dev/null
    wait $LLAMA_PID $PROXY_PID 2>/dev/null
    echo "Done."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start llama-server in background
echo "Starting llama-server on :8001 [$MODE mode — $ALIAS]..."

# $LLAMA_SERVER \
#     -hf Qwen/Qwen3-32B-GGUF:Q4_K_M \
#     --alias "qwen3-32b" \
#     --n-gpu-layers 99 \
#     --flash-attn on \
#     --ctx-size $CONTEXT \
#     --cache-type-k q8_0 --cache-type-v q8_0 \
#     --parallel 1 \
#     --jinja \
#     --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 \
#     --port 8001 &
# LLAMA_PID=$!

# $LLAMA_SERVER \
#     --model "./models/qwen3-32b/Qwen_Qwen3-32B-Q4_K_M.gguf" \
#     --alias "qwen3-32b" \
#     --n-gpu-layers 99 \
#     --flash-attn on \
#     --ctx-size $CONTEXT \
#     --cache-type-k q8_0 --cache-type-v q8_0 \
#     --parallel 1 \
#     --jinja \
#     --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 \
#     --port 8001 &
# LLAMA_PID=$!

$LLAMA_SERVER \
    --model "$MODEL" \
    --alias "$ALIAS" \
    --n-gpu-layers 99 \
    --flash-attn on \
    "${EXTRA_ARGS[@]}" \
    --ctx-size $CONTEXT \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --parallel 1 \
    --jinja \
    "${SAMPLING_ARGS[@]}" \
    --port 8001 &
LLAMA_PID=$!

# Wait a moment for llama-server to start
sleep 3

# Start proxy inspector in background
echo "Starting proxy inspector on :9001 (UI on :9002)..."
python3 "$PROXY" --port 9001 --target http://localhost:8001 --ui-port 9002 &
PROXY_PID=$!

echo ""
echo "════════════════════════════════════════════"
echo "  llama-server:  http://localhost:8001"
echo "  proxy:         http://localhost:9001"
echo "  inspector UI:  http://localhost:9002"
echo "  Press Ctrl+C to stop both"
echo "════════════════════════════════════════════"
echo ""

# Wait for either to exit
wait -n $LLAMA_PID $PROXY_PID
cleanup
