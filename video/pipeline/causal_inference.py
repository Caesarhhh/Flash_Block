# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional
import torch
import os
import time
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from vis_tensor import visualize_bool_tensor
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation, log_gpu_memory
from utils.debug_option import DEBUG
import torch.distributed as dist

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def local_nxn_match_copy_with_cosmap(
    denoised_pred: torch.Tensor,        # [B,C,T,H,W]
    self_denoised_pred: torch.Tensor,   # [B,C,T,H,W]
    n: int = 3,                         # 邻域大小 n×n（建议奇数）
    padding_mode: str = "replicate",    # "replicate" / "reflect" / "constant"
    eps: float = 1e-6,
):
    """
    对每个空间位置 (h,w)，在 self_denoised_pred 的 n×n 邻域里找与 denoised_pred 最相似的点（cosine）。
    匹配特征维度：D = C*T（将通道和时间展平后做 cosine 匹配）
    
    返回:
      out:     [B,C,T,H,W]  （新tensor，不修改 self_denoised_pred）
      idx:     [B,H,W]      （0..n^2-1，unfold 行优先从左上到右下）
      cos_sim: [B,C,H,W]    （每通道沿 T 做 cosine：cos(a[b,c,:,h,w], out[b,c,:,h,w])）
    """
    assert denoised_pred.shape == self_denoised_pred.shape
    assert n >= 1, "n must be >= 1"
    B, C, T, H, W = denoised_pred.shape

    # 如果 n 是偶数，也能跑，但“中心”不对称；通常建议奇数
    pad = n // 2
    D = C * T

    # 1) 匹配特征：展平 (C,T) -> [B, D, H, W]
    q = denoised_pred.reshape(B, D, H, W)
    x = self_denoised_pred.reshape(B, D, H, W)

    # 2) unfold n×n 邻域: [B, D*n*n, H*W] -> [B, D, K, HW]
    K = n * n
    x_pad = F.pad(x, (pad, pad, pad, pad), mode=padding_mode)
    neigh = F.unfold(x_pad, kernel_size=n, stride=1).view(B, D, K, H * W)

    qv = q.view(B, D, H * W)  # [B, D, HW]

    # 3) cosine 相似度并取 argmax: sim [B, K, HW]
    qn = F.normalize(qv, dim=1, eps=eps)
    nn = F.normalize(neigh, dim=1, eps=eps)
    sim = (qn.unsqueeze(2) * nn).sum(dim=1)  # [B, K, HW]
    best = sim.argmax(dim=1)                 # [B, HW] in 0..K-1

    # 4) gather 最优候选 -> out [B,C,T,H,W]
    gather_idx = best.view(B, 1, 1, H * W).expand(B, D, 1, H * W)  # [B,D,1,HW]
    out_flat = neigh.gather(dim=2, index=gather_idx).squeeze(2)    # [B,D,HW]
    out = out_flat.view(B, C, T, H, W).contiguous()

    # 5) cos_sim map: [B,C,H,W]（每通道沿 T）
    a = denoised_pred
    b = out
    dot = (a * b).sum(dim=2)  # [B,C,H,W]
    na = torch.sqrt((a * a).sum(dim=2).clamp_min(eps))
    nb = torch.sqrt((b * b).sum(dim=2).clamp_min(eps))
    cos_sim = dot / (na * nb)

    idx = best.view(B, H, W)
    return out, idx, cos_sim



class CausalInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        if DEBUG:
            print(f"args.model_kwargs: {args.model_kwargs}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # hard code for Wan2.1-T2V-1.3B
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = args.model_kwargs.local_attn_size

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        sparsity_params= None
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_output_frames, num_channels, height, width = noise.shape
        assert num_output_frames % self.num_frame_per_block == 0
        num_blocks = num_output_frames // self.num_frame_per_block

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )

        if sparsity_params is None:
            sparsity_params = {
                "is_calibrate": False,
                "is_sparsity": False,
                "use_cache": False,
                "use_svg2": False,
                "sparge_attn": False,
                "search_range": 3,
                "use_pab": False,
            }

        if hasattr(self.generator, "model") and hasattr(self.generator.model, "reset_pab_state"):
            self.generator.model.reset_pab_state()

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            vae_start = torch.cuda.Event(enable_timing=True)
            vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        local_attn_cfg = getattr(self.args.model_kwargs, "local_attn_size", -1)
        kv_policy = ""
        if local_attn_cfg != -1:
            # local attention
            kv_cache_size = local_attn_cfg * self.frame_seq_length
            kv_policy = f"int->local, size={local_attn_cfg}"
        else:
            # global attention
            kv_cache_size = num_output_frames * self.frame_seq_length
            kv_policy = "global (-1)"
        # print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self.generator.model.local_attn_size = self.local_attn_size
        # print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
        self._set_all_modules_max_attention_size(self.local_attn_size)

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 2: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        att_time_alls = []
        stttt=time.time()
        selected_index=None
        unipatch_selected_index=None
        cos_sim=None
        selected_indexs = []
        for frame_idx,current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()
            selected_index=None
            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]

            # [Optim] Step 0 Acceleration: Inherit mask from previous frame WITH DILATION
            # if frame_idx > 0 and getattr(self, 'cached_mask', None) is not None:
            #     # self.cached_mask is [B, F, H_small, W_small] boolean
            #     prev_mask = self.cached_mask.float()
            #     # Reshape for 2D MaxPool: [B*F, 1, H, W]
            #     bf = prev_mask.shape[0]
            #     h_m, w_m = prev_mask.shape[1], prev_mask.shape[2]
            #     prev_mask_reshaped = prev_mask.view(bf, 1, h_m, w_m)
            #   
            #     # Dilate
            #     dilated_mask = torch.nn.functional.max_pool2d(prev_mask_reshaped, kernel_size=3, stride=1, padding=1)
            #   
            #     # Flatten
            #     selected_index = (dilated_mask > 0.5).flatten()
            #   
            #     # Update unipatch index for masking logic
            #     # Assuming hardcoded shape 3, 30, 52 matches the model config (risky but matches existing code)
            #     # If cached_mask shape is known, we can prefer that.
            #     # prev_mask is [B, F, 30, 52]
            #     unipatch_selected_index = selected_index.view(prev_mask.shape[0], h_m, w_m).repeat_interleave(2, -1).repeat_interleave(2, -2)
            denoised_pred_=None
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"current_timestep: {current_timestep}")

                # set current timestep
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    st=time.time()
                    try:
                        from svg.timer import operator_log_data
                        operator_log_data.clear()
                    except Exception:
                        pass
                    if sparsity_params is not None:
                        sparsity_params["denoising_step"]=index
                        sparsity_params["frame_idx"]=frame_idx
                    _, denoised_pred, att_time_all = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        sparsity_params=sparsity_params,
                        selected_index=selected_index
                    )
                    if sparsity_params is not None and sparsity_params.get("use_selected_index", False):
                        if selected_index is not None:
                            mask = ~unipatch_selected_index[None, :, None, :, :]  # same thing

                            denoised_pred = torch.where(
                                mask,
                                aligned_prev,
                                denoised_pred
                            )
                        if sparsity_params["denoising_step"] == 0:
                            search_range = sparsity_params.get("search_range", 0) if sparsity_params is not None else 0
                            aligned_prev = self.denoised_pred
                            if search_range > 0:
                                aligned_prev, _, cos_sim = local_nxn_match_copy_with_cosmap(self.denoised_pred, denoised_pred, search_range)
                                # aligned_prev = torch.gather(raw_patches, 2, idx_expanded).squeeze(2).view(b, f, c, h, w)
                            else:
                                cos_sim = torch.nn.functional.cosine_similarity(denoised_pred,self.denoised_pred,dim=2)
                            selected_index = (-torch.nn.functional.max_pool2d(
                                    -cos_sim, kernel_size=2, stride=2
                                )<0.92)
                            # selected_index=torch.nn.functional.min_pool2d(
                            #         selecte_index.float(),
                            #         kernel_size=3,
                            #         stride=1,
                            #      0   padding=1
                            #     )>
                            selected_index=selected_index.flatten()
                            print(selected_index.float().mean())
                            
                            # Cache the mask (boolean tensor relative to coarse grid) BEFORE flattening logic lost dimensions?
                            # Actually L237 result was: (-cos_sim etc) < 0.9.
                            # We can reconstruct or cache the result of L237 logic.
                            # But selected_index at L237 is local.
                            # Let's use the reshaped view of selected_index to save cache.
                            # Hardcoded dimensions from L246: 3, 30, 52
                            self.cached_mask = selected_index.view(cos_sim.shape[1], cos_sim.shape[2] // 2, cos_sim.shape[3] // 2)
                            
                            unipatch_selected_index=selected_index.view(cos_sim.shape[1], cos_sim.shape[2] // 2, cos_sim.shape[3] // 2).repeat_interleave(2, -1).repeat_interleave(2, -2)
                    att_time_alls+=[att_time_all]
                    # print(time.time()-st)
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                    denoised_pred_=denoised_pred
                else:
                    # for getting real output
                                
                    # [Optim] Global Latent Alignment (Fix Final Step Artifacts)
                    # Align the statistics (mean) of the cached background to match the newly computed foreground.
                    # This prevents global lighting/tone shifts from creating seams.
                    # _selected_index=None
                    # if cos_sim is not None and frame_idx > 0:
                    #     _selected_index = (-torch.nn.functional.max_pool2d(
                    #             -cos_sim, kernel_size=2, stride=2
                    #         )<0.98)
                    #     _selected_index=_selected_index.flatten()
                    #     print(_selected_index.float().mean())
                    #     unipatch_selected_index=_selected_index.view(cos_sim.shape[1], cos_sim.shape[2] // 2, cos_sim.shape[3] // 2).repeat_interleave(2, -1).repeat_interleave(2, -2)

                    sparsity_params["denoising_step"]=index
                    sparsity_params["frame_idx"]=frame_idx
                    _, denoised_pred, att_time_all = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                        sparsity_params=sparsity_params,
                        # selected_index=selected_index
                    )
                    # if selected_index is not None:
                    #     mask = ~unipatch_selected_index[None, :, None, :, :]  # same thing
                    #     denoised_pred = torch.where(
                    #         mask,
                    #         aligned_prev,
                    #         denoised_pred
                    #     )
                    att_time_alls+=[att_time_all]
            # Step 2.2: record the model's output
            # if frame_idx>1:
            #     cos_sim=torch.nn.functional.cosine_similarity(denoised_pred,self.denoised_pred,dim=2)
            #     denoised_pred[(cos_sim>0.95)[:,:,None].repeat(1,1,16,1,1)]=self.denoised_pred[(cos_sim>0.95)[:,:,None].repeat(1,1,16,1,1)]
            if selected_index is not None:
                selected_indexs.append(selected_index.view(cos_sim.shape[1], cos_sim.shape[2] // 2, cos_sim.shape[3] // 2))
            self.denoised_pred=denoised_pred_[:,-1:].repeat(1,3,1,1,1)
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)
            # Step 2.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            sparsity_params["denoising_step"]=index+1
            sparsity_params["frame_idx"]=frame_idx
            _,_,att_time_all=self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                sparsity_params=sparsity_params,
                selected_index=selected_index
            )
            att_time_alls+=[att_time_all]
            # print(f"{sum(att_time_alls[-5:])} {time.time()-stttt}")

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 3: Decode the output
        video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video.mul_(0.5).add_(0.5).clamp_(0, 1)
        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")
        if len(selected_indexs)>0:
            selected_indexs = torch.stack(selected_indexs, dim=0)
        att_time_alls=sum(att_time_alls)
        if return_latents:
            return video, output.to(noise.device), att_time_alls
        else:
            return video, att_time_alls

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass
