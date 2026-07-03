#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export MODEL="${MODEL:-$ROOT/models/TraDo-8B-Thinking}"
export BLOCK_SIZE="${BLOCK_SIZE:-8}"
exec "$SCRIPT_DIR/eval.sh" "$@"
