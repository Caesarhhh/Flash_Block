# FlashBlock dLLM

This directory contains the minimal dLLM code path used for FlashBlock evaluation
and long-context rolling throughput tests. The shared FlashAttention fork is
kept at `../flash-attention` and is used by both `video/` and `dllm/`.

## Build

```bash
cd dllm
bash scripts/build_flash_attn.sh
```

The dLLM path uses the optimized FlashBlock history kernel by default.

## Evaluation

Download or place the TraDo weights in a local directory before running the
custom dLLM engine. The scripts default to `dllm/models/TraDo-8B-Thinking`;
you can also point `MODEL` to any local checkpoint directory that contains the
model `.safetensors` files.

```bash
cd dllm
mkdir -p models
huggingface-cli download Gen-Verse/TraDo-8B-Thinking \
  --local-dir models/TraDo-8B-Thinking
```

```bash
cd dllm
MODEL=models/TraDo-8B-Thinking \
MODE=s2 \
BLOCK_SIZE=4 \
DYNAMIC_THRESHOLD=0.9 \
EVAL_LIMIT=128 \
bash scripts/eval.sh
```

For a quick smoke test, add `EVAL_LIMIT=1 MAX_TOKEN=64`.

Set `EVAL_DATASETS` to run multiple datasets, one per line:

```bash
EVAL_DATASETS=$'AIME2024 math\nMATH500 math\nGSM8K math\nLiveCodeBench code\nLiveBench math' \
bash scripts/eval.sh
```

`scripts/eval.sh` only accepts these five evaluation datasets. During generation the
engine shows a progress bar only; throughput and latency profiling are reserved for the
rolling speed test script.

## Rolling Speed Test

The rolling script probes the maximum KV-cache capacity on the current machine by
default, then rolls from `CHECKPOINT_START` to the largest safe checkpoint.

```bash
cd dllm
MODEL=models/TraDo-8B-Thinking \
MODES="baseline s2" \
DATASET=LiveCodeBench \
DATA_TYPE=code \
CHECKPOINT_START=100000 \
bash scripts/rolling_speedup.sh
```

To force a fixed range instead of rolling to the detected maximum, set
`CHECKPOINT_END`, for example `CHECKPOINT_END=800000`.

To only print the KV-cache capacity without running the benchmark:

```bash
PROBE_ONLY=1 bash scripts/rolling_speedup.sh
```

Large datasets such as LiveCodeBench and LiveBench are not vendored here. Put their JSON
files under `dllm/data/` before running the full evaluation.

## Further Performance Improvement

You can optionally train the dLLM policy on generated rollout trajectories. First prepare
dense rollout trajectories, then train from the saved `.pt` file:

```bash
cd dllm
MODEL=models/TraDo-8B-Thinking \
OUTPUT=data/train_trajectories.pt \
BATCH_SIZE=32 \
bash scripts/prepare_trajectories.sh
```

```bash
cd dllm
MODEL=models/TraDo-8B-Thinking \
ROLLOUT_DATA=data/train_trajectories.pt \
bash scripts/train.sh
```

The training script starts from the base model by default. Set
`train.decoder_resume_path` in the training config to resume from a relative LoRA
checkpoint path such as `checkpoints/run_name/ddt_test/step.pt`.
