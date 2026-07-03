#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
MAX_JOBS="${MAX_JOBS:-2}" pip install -e . -v --no-build-isolation
