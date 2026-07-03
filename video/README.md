# FlashBlock Video Inference

This folder contains the video-generation part of FlashBlock. It is a cleaned inference
tree based on LongLive/Wan with only:

- baseline causal video generation;
- FlashBlock reuse-a calibration and inference;
- optional attention-core profiling.

The local FlashAttention fork is required for FlashBlock because the attention API must
support `return_hist`, `block_size`, `head_mask`, `attn_output_past`, and `logsumexp`.

## Prepare Models

Place or symlink checkpoints as follows:

```text
video/
├── longlive_models/
│   └── models/
│       ├── longlive_base.pt
│       └── lora.pt
└── wan_models/
    └── Wan2.1-T2V-1.3B/
        ├── Wan2.1_VAE.pth
        ├── models_t5_umt5-xxl-enc-bf16.pth
        └── google/umt5-xxl/
```

The release tree does not include these weights.

## Environment

Use the shared environment setup in the repository root README:

```bash
cd ..
pip install -r requirements.txt
MAX_JOBS=2 bash flash-attention/build.sh
```

## Prompt File

`inference.py` reads a text file where each line is one prompt. A low-motion sample
prompt is included at `prompts/sample_static.txt`:

```text
A person is eating a chocolate truffle.
```

This example is a good sanity check for FlashBlock because the subject and camera motion
are modest. Very fast motion or large camera changes are generally more challenging for
aggressive reuse settings.

A VBench Material sample is also included at `prompts/sample_material.txt`:

```text
A clear glass of coffee is gently poured into a glass of milk.
```

## Baseline

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u inference.py \
  --config_path configs/longlive_inference_baseline.yaml \
  --data_path prompts/sample_static.txt \
  --output_folder outputs/baseline
```

## FlashBlock

`--reuse_a` enables FlashBlock. By default, video inference uses reuse steps `1,2,3,4`
with threshold `0.85`.

Default 120-frame video setting:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u inference.py \
  --config_path configs/longlive_inference_reuse_seed1.yaml \
  --data_path prompts/sample_static.txt \
  --output_folder outputs/flashblock_1234_th085 \
  --reuse_a \
  --calibration_samples 5
```

The command above is equivalent to explicitly passing:

```bash
--sparsity_steps 1,2,3,4 --head_sim_th 0.85
```

Higher-quality, less aggressive setting:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u inference.py \
  --config_path configs/longlive_inference_reuse_seed1.yaml \
  --data_path prompts/sample_static.txt \
  --output_folder outputs/flashblock_1234_th09 \
  --reuse_a \
  --sparsity_steps 1,2,3,4 \
  --head_sim_th 0.9 \
  --calibration_samples 5
```

Per-step thresholds are also supported:

```bash
--head_sim_th_by_step 1:0.85,2:0.85,3:0.85,4:0.95
```

## Speed Profiling

Add `--profile_attention_core` to print attention-core timing:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u inference.py \
  --config_path configs/longlive_inference_reuse_seed1.yaml \
  --data_path prompts/sample_static.txt \
  --output_folder outputs/flashblock_profile \
  --reuse_a \
  --calibration_samples 5 \
  --profile_attention_core
```

The log line has this form:

```text
[attention_core_profile] idx=0 total=12.1344s full_attn=4.4598s/1320 cur_attn=7.6745s/4680 merge=0.0000s/0
```

## Calibration

FlashBlock first runs a small calibration batch to estimate per-head similarity. The
default calibration prompts are embedded in `inference.py`; set `--calibration_samples`
to control how many are used.
