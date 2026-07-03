#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/flash-attention"

MAX_JOBS="${MAX_JOBS:-2}" pip install -e . -v --no-build-isolation
