#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASH_ROOT="$(cd "$ROOT/../flash-attention" && pwd)"

export PYTHONPATH="$FLASH_ROOT:$ROOT:$ROOT/sample:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PY="${PY:-python}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT="$("$PY" - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
  export MASTER_PORT
fi

CONFIG="${CONFIG:-$ROOT/configs/trado_eval_sparsity.yaml}"
MODEL="${MODEL:-$ROOT/models/TraDo-8B-Thinking}"
MODE="${MODE:-s2}"
TP="${TP:-2}"
MAX_ACTIVE="${MAX_ACTIVE:-50}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.9}"
MAX_TOKEN="${MAX_TOKEN:-32768}"
EVAL_LIMIT="${EVAL_LIMIT:-1280000}"
EVAL_SEED="${EVAL_SEED:-10085}"
OUTPUT_NAME="${OUTPUT_NAME:-flashblock_${MODE}_block${BLOCK_SIZE}_dyn${DYNAMIC_THRESHOLD}_limit${EVAL_LIMIT}}"

if [[ "${CHECK_FLASH_ATTN:-0}" == "1" ]]; then
"$PY" - "$FLASH_ROOT" <<'PY'
import os
import sys

flash_root = os.path.realpath(sys.argv[1])
try:
    import torch  # noqa: F401
    import flash_attn_2_cuda
except Exception as exc:
    raise SystemExit(
        "[eval] failed to import flash_attn_2_cuda; run "
        "`MAX_JOBS=2 bash flash-attention/build.sh` from the repository root first.\n"
        f"[eval] import error: {exc}"
    )

so_path = os.path.realpath(getattr(flash_attn_2_cuda, "__file__", ""))
if not so_path.startswith(flash_root + os.sep):
    raise SystemExit(
        "[eval] flash_attn_2_cuda is not loaded from this release tree.\n"
        f"[eval] expected under: {flash_root}\n"
        f"[eval] got: {so_path}\n"
        "[eval] rebuild with `MAX_JOBS=2 bash flash-attention/build.sh`."
    )
PY
fi

case "$MODE" in
  baseline|s0) SPARSITY_RATIO=0 ;;
  s[0-9]*) SPARSITY_RATIO="${MODE#s}" ;;
  *) SPARSITY_RATIO="$MODE" ;;
esac

if [[ -n "${EVAL_DATASETS:-}" ]]; then
  mapfile -t DATASETS <<< "$EVAL_DATASETS"
else
  DATASETS=(
    #"AIME2024 math"
    #"MATH500 math"
    #"GSM8K math"
    # "LiveCodeBench code"
    "LiveBench code"
  )
fi

for item in "${DATASETS[@]}"; do
  [[ -z "$item" ]] && continue
  set -- $item
  DATASET="$1"
  DATA_TYPE="$2"

  case "$DATASET $DATA_TYPE" in
    "AIME2024 math"|"MATH500 math"|"GSM8K math"|"LiveCodeBench code"|"LiveBench code") ;;
    *)
      echo "[eval] unsupported dataset/type: $DATASET $DATA_TYPE" >&2
      echo "[eval] supported: AIME2024 math, MATH500 math, GSM8K math, LiveCodeBench code, LiveBench code" >&2
      exit 2
      ;;
  esac

  if [[ ! -f "$ROOT/data/$DATASET.json" ]]; then
    echo "[eval] missing dataset file: $ROOT/data/$DATASET.json" >&2
    exit 3
  fi

  echo "[eval] dataset=$DATASET type=$DATA_TYPE mode=$MODE sparsity=$SPARSITY_RATIO block=$BLOCK_SIZE"
  (
    cd "$ROOT/sample"
    "$PY" trado_sample.py \
      config="$CONFIG" \
      seed="$EVAL_SEED" \
      model="$MODEL" \
      dataset.eval_dataset="$DATASET" \
      dataset.data_type="$DATA_TYPE" \
      dataset.limit="$EVAL_LIMIT" \
      outputname="$OUTPUT_NAME" \
      rollout.sparsity_ratio="$SPARSITY_RATIO" \
      rollout.block_size="$BLOCK_SIZE" \
      rollout.denoising_steps_per_block="$DENOISING_STEPS" \
      rollout.max_token="$MAX_TOKEN" \
      rollout.dynamic_threshold="$DYNAMIC_THRESHOLD" \
      rollout.max_active="$MAX_ACTIVE" \
      rollout.tensor_parallel_size="$TP"
  )

  if [[ "$DATA_TYPE" == "code" ]]; then
    (
      cd "$ROOT/reward"
      "$PY" execute.py \
        config="$CONFIG" \
        model="$MODEL" \
        dataset.eval_dataset="$DATASET" \
        dataset.data_type="$DATA_TYPE" \
        dataset.limit="$EVAL_LIMIT" \
        outputname="$OUTPUT_NAME"
    )
  fi

  (
    cd "$ROOT/reward"
    "$PY" reward.py \
      config="$CONFIG" \
      model="$MODEL" \
      dataset.eval_dataset="$DATASET" \
      dataset.data_type="$DATA_TYPE" \
      dataset.limit="$EVAL_LIMIT" \
      outputname="$OUTPUT_NAME"
  )
done
