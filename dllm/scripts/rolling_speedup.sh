#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLASH_ROOT="$(cd "$ROOT/../flash-attention" && pwd)"

export PYTHONPATH="$FLASH_ROOT:$ROOT:$ROOT/sample:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TRADO_SYNC_PROFILE="${TRADO_SYNC_PROFILE:-1}"
export TRADO_CONSISTENT_SAMPLING_PARAMS="${TRADO_CONSISTENT_SAMPLING_PARAMS:-1}"
export TRADO_WARMUP_MODEL_LEN="${TRADO_WARMUP_MODEL_LEN:-1024}"

PY="${PY:-python}"
MODEL="${MODEL:-$ROOT/models/TraDo-8B-Thinking}"
CONFIG="${CONFIG:-$ROOT/configs/trado_eval_sparsity.yaml}"
OUTDIR="${OUTDIR:-$ROOT/logs/rolling_$(date +%Y%m%d_%H%M%S)}"
MODES="${MODES:-baseline s2}"
if [[ -n "${TP:-}" ]]; then
  TP="$TP"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
  TP="${#_visible_gpus[@]}"
else
  TP="1"
fi
MAX_ACTIVE="${MAX_ACTIVE:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
DYNAMIC_THRESHOLD="${DYNAMIC_THRESHOLD:-0.9}"
DATASET="${DATASET:-LiveCodeBench}"
DATA_TYPE="${DATA_TYPE:-code}"
CHECKPOINT_START="${CHECKPOINT_START:-100000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100000}"
CHECKPOINT_WINDOW="${CHECKPOINT_WINDOW:-10000}"
CHECKPOINT_END="${CHECKPOINT_END:-}"
STOP_CONTEXT="${STOP_CONTEXT:-}"
PROBE_MAX_KV="${PROBE_MAX_KV:-1}"
PROBE_ONLY="${PROBE_ONLY:-0}"

mkdir -p "$OUTDIR"

if [[ "$PROBE_MAX_KV" == "1" || "$PROBE_ONLY" == "1" ]]; then
  echo "[rolling] probing KV cache capacity"
  PROBE_LOG_FILE="$OUTDIR/probe_kv_cache.log"
  "$PY" "$ROOT/scripts/probe_kv_cache_shape.py" \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --tensor-parallel-size "$TP" \
    --block-size "$BLOCK_SIZE" \
    --batch-size "$MAX_ACTIVE" 2>&1 | tee "$PROBE_LOG_FILE"
  CAPACITY="$(awk '/capacity_tokens_total_context/ {print $2}' "$PROBE_LOG_FILE" | tail -n 1)"
  if [[ -z "$CAPACITY" ]]; then
    echo "[rolling] failed to parse KV capacity" >&2
    exit 1
  fi
  if [[ "$PROBE_ONLY" == "1" ]]; then
    exit 0
  fi
  MAX_CHECKPOINT_END="$(( (CAPACITY - CHECKPOINT_WINDOW) / CHECKPOINT_INTERVAL * CHECKPOINT_INTERVAL ))"
  if (( MAX_CHECKPOINT_END < CHECKPOINT_START )); then
    MAX_CHECKPOINT_END="$CHECKPOINT_START"
  fi
  if [[ -z "$CHECKPOINT_END" || "$CHECKPOINT_END" -gt "$MAX_CHECKPOINT_END" ]]; then
    CHECKPOINT_END="$MAX_CHECKPOINT_END"
  fi
fi

CHECKPOINT_END="${CHECKPOINT_END:-800000}"
STOP_CONTEXT="${STOP_CONTEXT:-$(( CHECKPOINT_END + CHECKPOINT_WINDOW ))}"
echo "[rolling] using CHECKPOINT_END=$CHECKPOINT_END STOP_CONTEXT=$STOP_CONTEXT"

echo "[rolling] outdir=$OUTDIR modes=$MODES dataset=$DATASET tp=$TP max_active=$MAX_ACTIVE"

for mode in $MODES; do
  echo "[rolling] start mode=$mode"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  JETENGINE_TIME_LOG_PATH="$OUTDIR/${mode}_time_logs.pt" \
  "$PY" "$ROOT/scripts/trado_context_stage_benchmark.py" \
    --config "$CONFIG" \
    --model "$MODEL" \
    --output-dir "$OUTDIR" \
    --mode "$mode" \
    --max-active "$MAX_ACTIVE" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dynamic-threshold "$DYNAMIC_THRESHOLD" \
    --stop-context "$STOP_CONTEXT" \
    --checkpoint-start "$CHECKPOINT_START" \
    --checkpoint-end "$CHECKPOINT_END" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --checkpoint-window "$CHECKPOINT_WINDOW" \
    dataset.eval_dataset="$DATASET" \
    dataset.data_type="$DATA_TYPE" \
    rollout.block_size="$BLOCK_SIZE" \
    rollout.denoising_steps_per_block="$DENOISING_STEPS" \
    rollout.max_active="$MAX_ACTIVE" \
    rollout.tensor_parallel_size="$TP" \
    rollout.gpu_memory_utilization="$GPU_MEM_UTIL"
done

"$PY" - "$OUTDIR" <<'PY'
import csv
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {}
for path in sorted(out.glob("*_summary.csv")):
    mode = path.name.removesuffix("_summary.csv")
    with path.open() as f:
        rows[mode] = list(csv.DictReader(f))

base = rows.get("baseline") or rows.get("s0")
if not base:
    print(f"[rolling] summaries written to {out}")
    raise SystemExit

base_by_stage = {r["stage"]: r for r in base}
print("stage,mode,latency_speedup,tps_speedup,base_latency_ms,mode_latency_ms,base_tps,mode_tps")
for mode, mode_rows in rows.items():
    if mode in ("baseline", "s0"):
        continue
    for r in mode_rows:
        b = base_by_stage.get(r["stage"])
        if not b:
            continue
        b_lat = float(b["avg_latency_ms"])
        m_lat = float(r["avg_latency_ms"])
        b_tps = float(b["tps"])
        m_tps = float(r["tps"])
        if b_lat > 0 and m_lat > 0:
            print(f"{r['stage']},{mode},{b_lat/m_lat:.4f},{m_tps/b_tps:.4f},{b_lat:.4f},{m_lat:.4f},{b_tps:.4f},{m_tps:.4f}")
PY
