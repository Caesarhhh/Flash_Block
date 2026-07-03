#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASH_ROOT="$(cd "$ROOT/../flash-attention" && pwd)"

export PYTHONPATH="$FLASH_ROOT:$ROOT:$ROOT/sample:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_CONFIG="${CONFIG:-$ROOT/configs/train.yaml}"
RUN_CONFIG="$BASE_CONFIG"
TMP_CONFIG=""

if [[ -n "${MODEL:-}" || -n "${OUTPUT_ROOT:-}" || -n "${ROLLOUT_DATA:-}" || -n "${BATCH_SIZE:-}" || -n "${VAL_BATCH_SIZE:-}" || -n "${GRAD_ACCUM:-}" || -n "${LR:-}" || -n "${NUM_ITERS:-}" ]]; then
  TMP_PARENT="${TMPDIR:-$ROOT/.cache/temp}"
  mkdir -p "$TMP_PARENT"
  TMP_CONFIG="$(mktemp "$TMP_PARENT/flashblock_train.XXXXXX.yaml")"
  python - "$BASE_CONFIG" "$TMP_CONFIG" <<'PY'
import os
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
if os.environ.get("MODEL"):
    cfg.paths.model = os.environ["MODEL"]
if os.environ.get("OUTPUT_ROOT"):
    cfg.paths.experiment = os.environ["OUTPUT_ROOT"]
if os.environ.get("ROLLOUT_DATA"):
    cfg.paths.rollout_data = os.environ["ROLLOUT_DATA"]
if os.environ.get("BATCH_SIZE"):
    cfg.data.batch_size = int(os.environ["BATCH_SIZE"])
if os.environ.get("VAL_BATCH_SIZE"):
    cfg.data.val_batch_size = int(os.environ["VAL_BATCH_SIZE"])
if os.environ.get("GRAD_ACCUM"):
    cfg.train.gradient_accumulation_steps = int(os.environ["GRAD_ACCUM"])
if os.environ.get("LR"):
    cfg.train.lr = float(os.environ["LR"])
if os.environ.get("NUM_ITERS"):
    cfg.train.num_iters = int(os.environ["NUM_ITERS"])
OmegaConf.save(cfg, sys.argv[2])
PY
  RUN_CONFIG="$TMP_CONFIG"
fi

cleanup() {
  [[ -n "$TMP_CONFIG" && -f "$TMP_CONFIG" ]] && rm -f "$TMP_CONFIG"
}
trap cleanup EXIT

cd "$ROOT"
python train.py --config "$RUN_CONFIG"
