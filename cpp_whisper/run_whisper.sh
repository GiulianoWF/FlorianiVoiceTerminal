#!/bin/bash
# Wrapper script for whisper_stt.
# Locates the model from Bazel runfiles and forwards all args to the binary.
set -e

RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"
MODEL="${RUNFILES}/_main/models/selected_model.bin"
BINARY="${RUNFILES}/_main/whisper_stt"

exec "${BINARY}" -m "${MODEL}" "$@"
