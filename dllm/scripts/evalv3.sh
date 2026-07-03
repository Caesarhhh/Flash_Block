#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASH_ROOT="$(cd "$ROOT/../flash-attention" && pwd)"

export PYTHONPATH="$FLASH_ROOT:$ROOT:$ROOT/sample:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-$((10000 + RANDOM % 50000))}"

CONFIG="${CONFIG:-$ROOT/configs/trado_eval_sparsity.yaml}"
MODEL="${MODEL:-models/TraDo-8B-Thinking}"
MODE="${MODE:-s2}"
TP="${TP:-2}"
MAX_ACTIVE="${MAX_ACTIVE:-50}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.9}"
EVAL_LIMIT="${EVAL_LIMIT:-128000}"
EVAL_SEED="${EVAL_SEED:-10085}"
OUTPUT_NAME="${OUTPUT_NAME:-3100flashblock_${MODE}_block${BLOCK_SIZE}_dyn${DYNAMIC_THRESHOLD}_limit${EVAL_LIMIT}}"

case "$MODE" in
  baseline|s0) SPARSITY_RATIO=0 ;;
  s[0-9]*) SPARSITY_RATIO="${MODE#s}" ;;
  *) SPARSITY_RATIO="$MODE" ;;
esac

if [[ -n "${EVAL_DATASETS:-}" ]]; then
  mapfile -t DATASETS <<< "$EVAL_DATASETS"
else
  DATASETS=(
    # "AIME2024 math"
    "MATH500 math"
    "GSM8K math"
    "LiveCodeBench code"
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
    python trado_sample.py \
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
      rollout.dynamic_threshold="$DYNAMIC_THRESHOLD" \
      rollout.max_active="$MAX_ACTIVE" \
      rollout.tensor_parallel_size="$TP"
  )

  if [[ "$DATA_TYPE" == "code" ]]; then
    (
      cd "$ROOT/reward"
      python execute.py \
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
    python reward.py \
      config="$CONFIG" \
      model="$MODEL" \
      dataset.eval_dataset="$DATASET" \
      dataset.data_type="$DATA_TYPE" \
      dataset.limit="$EVAL_LIMIT" \
      outputname="$OUTPUT_NAME"
  )
done
