#!/bin/bash
# Wrapper script for voice_query.
# Paths are passed as positional args via rlocationpath in BUILD.bazel.
set -e

# --- Runfiles resolution (Bazel idiom) ---
# shellcheck disable=SC1090
source "${RUNFILES_DIR:-"${BASH_SOURCE[0]}.runfiles"}/_repo_mapping" 2>/dev/null || true
RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"

rlocation() { echo "${RUNFILES}/$1"; }

BINARY="$(rlocation "$1")"
WHISPER_MODEL="$(rlocation "$2")"
KOKORO_MODEL="$(rlocation "$3")"
KOKORO_VOICE="$(rlocation "$4")"
shift 4

exec "${BINARY}" \
    --whisper-model "${WHISPER_MODEL}" \
    --kokoro-model "${KOKORO_MODEL}" \
    --kokoro-voice "${KOKORO_VOICE}" \
    --gpu \
    "$@"
