<div align="center">

# FlashBlock

### Attention Caching for Efficient Long-Context Block Diffusion

**ICML 2026**

[Project Page](https://caesarhhh.github.io/FlashBlock/) · [Video Code](video/README.md) · [dLLM Code](dllm/README.md)

</div>

FlashBlock accelerates long-context block diffusion by caching and reusing historical
attention outputs at the block level. The method is designed for generation workloads
where each block repeatedly attends to a long history: instead of recomputing the full
history for every selected denoising step and head, FlashBlock computes the current-block
attention and merges it with cached historical attention statistics.

This release contains both the video-generation code path and the diffusion language
model code path. Both paths use the shared FlashAttention fork under
`flash-attention/`.

## Highlights

- **Block-level attention caching:** cache historical attention outputs and log-sum-exp
  statistics for long-context block diffusion.
- **Head-wise adaptive reuse:** calibrate per-head similarity and reuse only selected
  denoising steps/heads.
- **Video and dLLM coverage:** the paper studies both video generation and diffusion
  language models; this repository is organized accordingly.

## Video Samples

The default video setting in this release uses `reuse_a`, denoising steps `1,2,3,4`,
threshold `0.85`, and 5 calibration prompts. The qualitative samples below are
baseline/FlashBlock comparisons selected from our runs, with relatively low-motion
examples emphasized because aggressive attention reuse is most stable when scene content
changes smoothly.

| Prompt | Baseline | FlashBlock |
| --- | --- | --- |
| A person is eating a chocolate truffle. | <video src="assets/video_samples/chocolate_truffle_baseline.mp4" controls muted loop width="260"></video> | <video src="assets/video_samples/chocolate_truffle_flashblock.mp4" controls muted loop width="260"></video> |
| A flower changes from purple to orange. | <video src="assets/video_samples/flower_color_change_baseline.mp4" controls muted loop width="260"></video> | <video src="assets/video_samples/flower_color_change_flashblock.mp4" controls muted loop width="260"></video> |
| A clear glass of coffee is gently poured into a glass of milk. | <video src="assets/video_samples/coffee_milk_baseline.mp4" controls muted loop width="260"></video> | <video src="assets/video_samples/coffee_milk_flashblock.mp4" controls muted loop width="260"></video> |

If your Markdown viewer does not render HTML video tags, open the files directly under
`assets/video_samples/`:

- Chocolate truffle: [baseline](assets/video_samples/chocolate_truffle_baseline.mp4) · [FlashBlock](assets/video_samples/chocolate_truffle_flashblock.mp4)
- Flower color change: [baseline](assets/video_samples/flower_color_change_baseline.mp4) · [FlashBlock](assets/video_samples/flower_color_change_flashblock.mp4)
- Coffee poured into milk: [baseline](assets/video_samples/coffee_milk_baseline.mp4) · [FlashBlock](assets/video_samples/coffee_milk_flashblock.mp4)

## Example Speed Snapshot

The video-generation speed numbers below are reported in the paper for
LongLive-1.3B on VBench2 with the FlashBlock video setting.

| Stage | Dense baseline | FlashBlock | Speedup |
| --- | ---: | ---: | ---: |
| Attention time | 19.89 s | 12.46 s | 1.60x |
| End-to-end video generation | 73.28 s | 66.07 s | 1.11x |

The end-to-end speedup is bounded by non-attention components in the LongLive video
pipeline, while FlashBlock directly targets the attention computation and KV-cache
access. Runtime still depends on GPU, FlashAttention build, prompt, and video length.

## Repository Layout

```text
flashblock/
├── README.md
├── requirements.txt
├── assets/
│   ├── paper/
│   └── video_samples/
├── dllm/
│   ├── README.md
│   ├── configs/
│   ├── data/
│   ├── models/
│   ├── reward/
│   ├── sample/
│   ├── scripts/
│   ├── train.py
│   └── utils/
├── flash-attention/
└── video/
    ├── README.md
    ├── inference.py
    ├── configs/
    ├── pipeline/
    ├── utils/
    └── wan/
```

## Environment Setup

The video and dLLM code paths share the same Python environment and the local
FlashAttention fork under `flash-attention/`. A minimal setup is:

```bash
cd flashblock

conda create -n flashblock python=3.10 -y
conda activate flashblock

python -m pip install --upgrade pip setuptools wheel

# Pick the PyTorch wheel that matches your CUDA driver/runtime.
# CUDA 12.1 is a common choice for A100/H100/H200 machines.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

If your environment does not already provide `flashinfer`, install the FlashInfer wheel
that matches your PyTorch/CUDA combination following the FlashInfer installation guide.
The dLLM model code imports `flashinfer`.

Build the bundled FlashAttention extension after installing the Python dependencies:

```bash
cd flashblock
MAX_JOBS=2 bash flash-attention/build.sh
```

Sanity checks:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("torch:", torch.__version__)
import flash_attn_2_cuda
print("flash_attn_2_cuda import ok")
PY
```

## Quick Start: Video

See [video/README.md](video/README.md) for checkpoint placement and complete commands.
The default FlashBlock command is:

```bash
cd video

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python -u inference.py \
  --config_path configs/longlive_inference_reuse_seed1.yaml \
  --data_path prompts/sample_static.txt \
  --output_folder outputs/flashblock_1234_th085 \
  --reuse_a \
  --calibration_samples 5
```

The command above defaults to:

```bash
--sparsity_steps 1,2,3,4 --head_sim_th 0.85
```

## Notes

- The video release keeps only the baseline path and the FlashBlock reuse-attention path.
- The custom FlashAttention operator is vendored once under `flash-attention/` and is
  shared by the video and dLLM paths.
- Model weights and generated outputs are intentionally not included. For dLLM
  evaluation and training, download the TraDo weights locally and pass the local
  directory through `MODEL=...`; see [dllm/README.md](dllm/README.md).

## Citation

```bibtex
@inproceedings{flashblock2026,
  title = {FlashBlock: Attention Caching for Efficient Long-Context Block Diffusion},
  booktitle = {International Conference on Machine Learning},
  year = {2026}
}
```
