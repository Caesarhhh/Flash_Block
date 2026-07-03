# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import importlib.util as importlib_util
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import time

if os.environ.get("LONGLIVE_DISABLE_XFORMERS", "1") == "1":
    _real_find_spec = importlib_util.find_spec

    def _find_spec_without_xformers(name, *args, **kwargs):
        if name == "xformers" or name == "sklearn" or name.startswith("sklearn."):
            return None
        return _real_find_spec(name, *args, **kwargs)

    importlib_util.find_spec = _find_spec_without_xformers

from pipeline import (
    CausalInferencePipeline,
)
from utils.dataset import TextDataset
from utils.misc import set_seed

from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--output_folder", type=str,  default=None, help="output_folder")
parser.add_argument("--data_path", type=str,  default=None, help="data_path")
parser.add_argument("--use_cache", action="store_true")
parser.add_argument("--reuse_a", action="store_true")
parser.add_argument("--sparsity_steps", type=str, default="1,2,3,4", help="Comma-separated denoising step indices for reuse_a, e.g. 1,2,3,4")
parser.add_argument("--head_sim_th", type=float, default=0.85, help="Head similarity threshold for reuse_a calibration")
parser.add_argument("--head_sim_th_by_step", type=str, default="", help="Optional per-step reuse_a thresholds, e.g. 1:0.8,2:0.8,3:0.8,4:0.9")
parser.add_argument("--head_lse_th", type=float, default=0.05, help="Mean per-head history logsumexp drift threshold for reuse_a calibration")
parser.add_argument("--calibration_samples", type=int, default=5, help="Number of videos used to average reuse_a head similarity calibration")
parser.add_argument("--profile_attention_core", action="store_true", help="Profile only attention kernel and reuse merge inside the model")
args = parser.parse_args()


def _write_video(path, frames, fps=16):
    if torch.is_tensor(frames):
        frames = frames.detach().cpu()
        if frames.dtype.is_floating_point:
            frames = frames.clamp(0, 255).to(torch.uint8)
    from torchvision.io import write_video
    write_video(path, frames, fps=fps)

config = OmegaConf.load(args.config_path)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single GPU mode on device {device}")

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)


# Initialize pipeline
# Note: checkpoint loading is now handled inside the pipeline __init__ method
print("[init] building pipeline")
pipeline = CausalInferencePipeline(config, device=device)
print("[init] pipeline ready")

# Load generator checkpoint
if config.generator_ckpt:
    print(f"[init] loading generator checkpoint: {config.generator_ckpt}")
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)
    print("[init] generator checkpoint loaded")

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # 在加载基础权重后，对 generator 的 transformer 模型应用 LoRA 包装
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # 加载 LoRA 权重（如果提供了 lora_ckpt）
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # 兼容包含 `generator_lora` 键或直接是 LoRA state dict 两种格式
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True
    print("[init] lora setup done")


# Move pipeline to appropriate dtype and device
print("[init] moving pipeline to dtype/device")
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)
print("[init] pipeline moved to device")

if args.data_path is not None:
    config.data_path = args.data_path
extended_prompt_path = config.data_path
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
print(f"[init] dataset ready: {config.data_path}")
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)
if args.output_folder is not None:
    config.output_folder = args.output_folder
# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

def parse_int_list(raw_value: str):
    values = [int(v.strip()) for v in raw_value.split(",") if v.strip()]
    return values or None


def parse_step_float_map(raw_value: str):
    result = {}
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        step, value = item.split(":", 1)
        result[int(step.strip())] = float(value.strip())
    return result

sparsity_params={
    "sparsity_steps": parse_int_list(args.sparsity_steps) or [],
    "is_sparsity": False,
    "head_sim_th": args.head_sim_th,
    "head_sim_th_by_step": parse_step_float_map(args.head_sim_th_by_step),
    "head_lse_th": args.head_lse_th,
    "profile_attention_core": args.profile_attention_core,
}

def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output

print("calibration start.")

sparsity_params["is_calibrate"] = True
if args.reuse_a:
    for block in pipeline.generator.model.blocks:
        self_attn = block.self_attn
        self_attn.head_mask.clear()
        self_attn.coses.clear()
        self_attn.coses_in.clear()
        self_attn.lse_deltas.clear()
        if hasattr(self_attn, "cos_sums"):
            self_attn.cos_sums.clear()
            self_attn.cos_counts.clear()
            self_attn.cos_in_sums.clear()
            self_attn.cos_in_counts.clear()
            self_attn.lse_delta_sums.clear()
            self_attn.lse_delta_counts.clear()
    calibration_prompts = [
        "A man walks in a street.",
        "The leaves gradually change from red to green.",
        "A man is doing yoga.",
        "A person is slurping noodles from a steaming bowl.",
        "A man is running.",
    ]
    num_calibration_samples = max(1, args.calibration_samples)
    prompts = [
        calibration_prompts[cal_idx % len(calibration_prompts)]
        for cal_idx in range(num_calibration_samples)
    ]
    sampled_noise = torch.randn(
        [num_calibration_samples, 21, 16, 60, 104], device=device, dtype=torch.bfloat16
    )
    sparsity_params["calibration_index"] = 0
    video, latents, _ = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        sparsity_params=sparsity_params
    )
    for block in pipeline.generator.model.blocks:
        self_attn = block.self_attn
        if not hasattr(self_attn, "cos_sums"):
            continue
        for step_index, cos_sum in self_attn.cos_sums.items():
            count = self_attn.cos_counts.get(step_index, 0)
            if count == 0:
                continue
            cos_mean = cos_sum / count
            head_sim_th = sparsity_params["head_sim_th_by_step"].get(step_index, sparsity_params["head_sim_th"])
            lse_mean = None
            if step_index in self_attn.lse_delta_sums:
                lse_count = self_attn.lse_delta_counts.get(step_index, 0)
                if lse_count > 0:
                    lse_mean = self_attn.lse_delta_sums[step_index] / lse_count
            self_attn.coses[step_index] = cos_mean
            if lse_mean is not None:
                self_attn.lse_deltas[step_index] = lse_mean
                self_attn.head_mask[step_index] = (cos_mean < head_sim_th) | (lse_mean > sparsity_params["head_lse_th"])
            else:
                self_attn.head_mask[step_index] = cos_mean < head_sim_th
        for step_index, cos_in_sum in self_attn.cos_in_sums.items():
            count = self_attn.cos_in_counts.get(step_index, 0)
            if count == 0:
                continue
            self_attn.coses_in[step_index] = cos_in_sum / count
sparsity_params["is_calibrate"]=False
sparsity_params["is_sparsity"] = args.reuse_a
sparsity_params["use_cache"] = args.use_cache
sparsity_params["search_range"] = 3

print("calibration done.")
latencys=[]
att_time_allses=[]

pbar=tqdm(enumerate(dataloader), disable=(local_rank != 0),ncols=20)
for i, batch_data in pbar:
    idx = batch_data['idx'].item()
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch
    prompt = batch['prompts'][0]
    if config.save_with_index:
        output_path = os.path.join(config.output_folder, f'{idx}-0_lora.mp4')
        output_path_ = os.path.join(config.output_folder, f'{idx}-1.mp4')
    else:
        output_path = os.path.join(config.output_folder, f'{prompt[:180]}-2.mp4')
        output_path_ = os.path.join(config.output_folder, f'{prompt[:180]}-2.mp4')
    # if os.path.exists(output_path) or os.path.exists(output_path_):
    #     continue

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames

    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    # print("sampled_noise.device", sampled_noise.device)
    # print("initial_latent.device", initial_latent.device)
    # print("prompts", prompts)
    # Generate 81 frames
    # print('sampled_noise.shape', sampled_noise.shape, 'prompts', prompts)
    # print('pipeline.generator', pipeline.generator)
    # print('pipeline.text_encoder', pipeline.text_encoder)
    # print('pipeline.vae', pipeline.vae)
    if args.profile_attention_core:
        sparsity_params["_attention_core_profile"] = {}
    st=time.time()
    video, latents, att_time_alls = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        low_memory=low_memory,
        profile=True,
        sparsity_params=sparsity_params
    )
    latencys.append(time.time()-st)
    att_time_allses.append(att_time_alls)
    if args.profile_attention_core and local_rank == 0:
        profile_stats = sparsity_params.get("_attention_core_profile", {})
        full_ms = profile_stats.get("full_attn_ms", 0.0)
        cur_ms = profile_stats.get("cur_attn_ms", 0.0)
        merge_ms = profile_stats.get("merge_ms", 0.0)
        total_ms = profile_stats.get("total_core_ms", full_ms + cur_ms + merge_ms)
        full_count = int(profile_stats.get("full_attn_count", 0))
        cur_count = int(profile_stats.get("cur_attn_count", 0))
        merge_count = int(profile_stats.get("merge_count", 0))
        print(
            "[attention_core_profile] "
            f"idx={idx} total={total_ms / 1000:.4f}s "
            f"full_attn={full_ms / 1000:.4f}s/{full_count} "
            f"cur_attn={cur_ms / 1000:.4f}s/{cur_count} "
            f"merge={merge_ms / 1000:.4f}s/{merge_count}"
        )
    pbar.set_description(f"latency - {sum(latencys)/len(latencys):.2f}s ; attn - {sum(att_time_allses)/len(att_time_allses)}")
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in range(config.num_samples):
            # All processes save their videos
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'{idx}-{seed_idx}_{model_type}.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'{prompt[:180]}-{seed_idx}.mp4')
            _write_video(output_path, video[seed_idx], fps=16)

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
if dist.is_initialized():
    dist.destroy_process_group()
