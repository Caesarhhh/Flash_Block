#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASH_ROOT="$(cd "$ROOT/../flash-attention" && pwd)"

export PYTHONPATH="$FLASH_ROOT:$ROOT:$ROOT/sample:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$ROOT"

ARGS=(
  scripts/prepare_trajectories.py
  --config "${CONFIG:-configs/train.yaml}"
  --model "${MODEL:-models/TraDo-8B-Thinking}"
  --output "${OUTPUT:-data/train_trajectories.pt}"
  --limit "${LIMIT:-0}"
  --batch-size "${BATCH_SIZE:-32}"
  --tensor-parallel-size "${TP:-1}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
  --block-size "${BLOCK_SIZE:-4}"
  --denoising-steps "${DENOISING_STEPS:-8}"
  --dynamic-threshold "${DYNAMIC_THRESHOLD:-0.9}"
  --max-tokens "${MAX_TOKENS:-8192}"
  --temperature "${TEMPERATURE:-1.0}"
  --top-k "${TOP_K:-0}"
  --top-p "${TOP_P:-1.0}"
  --remasking-strategy "${REMASKING_STRATEGY:-low_confidence_dynamic}"
  --seed "${SEED:-10085}"
)

if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi

python "${ARGS[@]}"
