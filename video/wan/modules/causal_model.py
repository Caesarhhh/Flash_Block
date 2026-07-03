# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
try:
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
except Exception:
    spas_sage2_attn_meansim_topk_cuda = None
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import time
import torch.distributed as dist
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

from utils.debug_option import DEBUG
try:
    from utils.pab_manager import PABConfig, PABManager
except Exception:
    PABConfig = None
    PABManager = None


def _profile_enabled(sparsity_params, tensor):
    return (
        sparsity_params is not None
        and sparsity_params.get("profile_attention_core", False)
        and not sparsity_params.get("is_calibrate", False)
        and tensor.is_cuda
    )


def _profile_add(sparsity_params, key, elapsed_ms):
    stats = sparsity_params.setdefault("_attention_core_profile", {})
    stats[key] = stats.get(key, 0.0) + elapsed_ms
    stats[key.replace("_ms", "_count")] = stats.get(key.replace("_ms", "_count"), 0) + 1
    stats["total_core_ms"] = stats.get("total_core_ms", 0.0) + elapsed_ms


def _event_pair():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _event_elapsed(start, end):
    end.synchronize()
    return start.elapsed_time(end)

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

import torch
import math

def compute_attention_prob(
    roped_query: torch.Tensor,
    k_cat: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
):
    """
    roped_query: [B, Q_len, H, D]
    k_cat:       [B, K_len, H, D]
    attn_mask:   broadcastable to [B, H, Q_len, K_len] (optional)

    Returns:
        attn_probs: [B, H, Q_len, K_len]
    """

    B, Q_len, H, D = roped_query.shape

    # → [B, H, L, D]
    Q = roped_query.permute(0, 2, 1, 3).contiguous()
    K = k_cat.permute(0, 2, 1, 3).contiguous()

    # scaled dot-product
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask

    # 数值稳定 softmax
    attn_probs = torch.softmax(attn_scores, dim=-1)

    return attn_probs

def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

def selected_causal_rope_apply(x, grid_sizes, freqs, start_frame=0,selected_index=None):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = x.shape[1]

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(f*h*w, 1, -1)
        if selected_index is not None:
            freqs_i = freqs_i[selected_index]
        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

import torch
import math

def compute_attention_score(
    roped_query: torch.Tensor,
    k_cat: torch.Tensor,
):
    """
    roped_query: [B, Q_len, H, D]
    k_cat:       [B, K_len, H, D]

    Returns:
        attn_scores: [B, H, Q_len, K_len]
    """

    B, Q_len, H, D = roped_query.shape
    _, K_len, _, _ = k_cat.shape

    # → [B, H, L, D]
    Q = roped_query.permute(0, 2, 1, 3).contiguous()
    K = k_cat.permute(0, 2, 1, 3).contiguous()

    # scaled dot product
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)

    return attn_scores

def pooled_attention_scores(q, k, grid_sizes, pool_size):
    """
    q: [B, Nq, nheads, dim]
    k: [B, Nk, nheads, dim]
    grid_sizes: tensor([[T, H, W]])  # for k
    pool_size: int

    return:
        attn: [B, heads, Nq, Kp]
        lse:  [B, heads, Nq]
    """

    B, Nq, nheads, dim = q.shape
    _, Nk, _, _ = k.shape

    T, H, W = grid_sizes[0].tolist()

    assert Nk == T * H * W

    # ---- reshape k to grid ----
    k = k.view(B, T, H, W, nheads, dim)

    # ---- pooling keys ----
    k = k.permute(0,1,4,5,2,3)   # B T heads dim H W
    k = k.reshape(B*T*nheads, dim, H, W)

    k_pool = torch.nn.functional.avg_pool2d(
        k,
        kernel_size=pool_size,
        stride=pool_size
    )

    Hp, Wp = k_pool.shape[-2:]

    k_pool = k_pool.view(B, T, nheads, dim, Hp, Wp)
    k_pool = k_pool.permute(0,1,4,5,2,3)  # B T Hp Wp heads dim
    k_pool = k_pool.reshape(B, T*Hp*Wp, nheads, dim)

    # ---- attention ----
    scale = dim ** -0.5

    scores = torch.einsum(
        "bqhd,bkhd->bhqk",
        q,
        k_pool
    ) * scale

    lse = torch.logsumexp(scores, dim=-1)

    attn = torch.softmax(scores, dim=-1)

    return attn, lse

def segmented_pooled_attention(q, k, grid_sizes, pool_size, n_segments):
    """
    return:
        attn_list: list[n] of attention
        lse_list: list[n] of lse
        seg_weight: [B, heads, Q, n_segments]
    """

    B, N, nheads, dim = k.shape

    seg_len = N // n_segments

    attn_list = []
    lse_list = []

    for i in range(n_segments):

        start = i * seg_len
        end = (i + 1) * seg_len if i < n_segments - 1 else N

        k_seg = k[:, start:end]

        # grid size for this segment
        T, H, W = grid_sizes[0].tolist()
        T = k.shape[1]//H//W

        # 假设沿 T 维切
        T_seg = (end - start) // (H * W)

        grid_seg = torch.tensor([[T_seg, H, W]], device=q.device)

        attn, lse = pooled_attention_scores(
            q,
            k_seg,
            grid_seg,
            pool_size
        )

        attn_list.append(attn)
        lse_list.append(lse)

    # stack lse
    lse_stack = torch.stack(lse_list, dim=-1)  # B heads Q n

    # segment softmax
    seg_weight = torch.softmax(lse_stack, dim=-1)

    return attn_list, lse_list, seg_weight
    
def mix_linear_attention(
    q,
    k,
    v,
    grid_sizes,
    pool_size,
    chunk_size,
    eps=1e-6,
):
    """
    q : [B, Nq, H, D]
    k : [B, Nk, H, D]
    v : [B, Nk, H, D]

    return
        out : [B, Nq, H, D]
    """

    B, Nq, H, D = q.shape
    Nk = k.shape[1]

    C = Nk // chunk_size

    # ------------------------------------------------
    # routing weights
    # ------------------------------------------------

    _, _, seg_weight = segmented_pooled_attention(
        q,
        k,
        grid_sizes,
        pool_size,
        C,
        num_frame_per_block=3,
    )
    # seg_weight : [B, H, Nq, C]

    # ------------------------------------------------
    # chunk KV
    # ------------------------------------------------

    k_chunk = k.view(B, C, chunk_size, H, D)
    v_chunk = v.view(B, C, chunk_size, H, D)

    k_chunk = k_chunk.permute(0,3,1,2,4)   # B,H,C,L,D
    v_chunk = v_chunk.permute(0,3,1,2,4)

    # ------------------------------------------------
    # linear attention statistics
    # ------------------------------------------------

    # S_c = Σ k v^T

    S = torch.einsum(
        "bhcld,bhcle->bhcde",
        k_chunk,
        v_chunk
    )  # [B,H,C,D,D]

    # Z_c = Σ k

    Z = k_chunk.sum(dim=3)  # [B,H,C,D]

    # ------------------------------------------------
    # apply Q to all chunks
    # ------------------------------------------------

    q_all = q.permute(0,2,1,3)  # [B,H,Nq,D]

    num = torch.einsum(
        "bhnd,bhcde->bhnce",
        q_all,
        S
    )  # [B,H,Nq,C,D]

    denom = torch.einsum(
        "bhnd,bhcd->bhnc",
        q_all,
        Z
    ).unsqueeze(-1) + eps

    chunk_out = num / denom  # [B,H,Nq,C,D]

    # ------------------------------------------------
    # apply routing weights
    # ------------------------------------------------

    seg_weight = seg_weight.unsqueeze(-1)  # [B,H,Nq,C,1]

    out = (chunk_out * seg_weight).sum(dim=3)

    out = out.permute(0,2,1,3)  # [B,Nq,H,D]

    return out

def mix_softmax_flash_attention(
    q,
    k,
    v,
    grid_sizes,
    pool_size,
    chunk_size,
    attention,
):
    """
    q,k,v : [B, N, H, D]

    return
        out : [B, N, H, D]
    """

    B, N, H, D = k.shape
    C = N // chunk_size

    # -------------------------------------------------
    # routing weights
    # -------------------------------------------------

    _, _, seg_weight = segmented_pooled_attention(
        q,
        k,
        grid_sizes,
        pool_size,
        C
    )

    # seg_weight
    # [B, H, N, C]

    # -------------------------------------------------
    # reshape KV into chunks
    # -------------------------------------------------

    k_chunk = k.view(B, C, chunk_size, H, D)
    v_chunk = v.view(B, C, chunk_size, H, D)

    # -------------------------------------------------
    # compute chunk attention outputs
    # -------------------------------------------------

    chunk_out_list = []

    for c in range(C):

        k_c = k_chunk[:, c]   # [B, chunk, H, D]
        v_c = v_chunk[:, c]

        out_c = attention(
            q,
            k_c,
            v_c,
        )  # [B, N, H, D]

        chunk_out_list.append(out_c)

    # stack

    chunk_out = torch.stack(chunk_out_list, dim=-2)

    # shape
    # [B, N, H, C, D]

    # -------------------------------------------------
    # apply routing
    # -------------------------------------------------

    seg_weight = seg_weight.permute(0,2,1,3)  # B,N,H,C
    seg_weight = seg_weight.unsqueeze(-1)

    out = (chunk_out * seg_weight).sum(dim=3)

    return out

class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6,
                 layer_i=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.head_mask = {}
        self.coses = {}
        self.coses_in = {}
        self.lse_deltas = {}
        self.cos_sums = {}
        self.cos_counts = {}
        self.cos_in_sums = {}
        self.cos_in_counts = {}
        self.lse_delta_sums = {}
        self.lse_delta_counts = {}
        self.processor=None
        self.k_cache = {}
        self.v_cache = {}
        self.cache_score = {}
        self.pab_self_count = {}
        self.pab_last_self = {}

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        sparsity_params=None,
        t=None,
        selected_index=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            act_local_start_index = None
            act_local_end_index = None
            frame_idx = None if sparsity_params is None else sparsity_params.get("frame_idx")
            denoising_step = None if sparsity_params is None else sparsity_params.get("denoising_step")
            use_pab = bool(PABManager is not None and sparsity_params is not None and sparsity_params.get("use_pab", False))
            pab_manager = None
            if use_pab:
                pab_manager = PABManager(
                    PABConfig(
                        self_broadcast=True,
                        self_threshold=sparsity_params.get("pab_self_threshold"),
                        self_range=sparsity_params.get("pab_self_range", 1),
                        self_reuse_steps=sparsity_params.get("pab_self_reuse_steps"),
                    )
                )
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            if selected_index is not None:
                roped_query = selected_causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame,selected_index=selected_index).type_as(v)
                roped_key = selected_causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame,selected_index=selected_index).type_as(v)
            else:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print("***********before attention***********")
            #     print(f"kv_cache_size = {kv_cache_size / frame_seqlen}")
            #     print(f"torch.is_grad_enabled() = {torch.is_grad_enabled()}")
            #     print(f"current_end = {current_end / frame_seqlen}")
            #     print(f"current_start = {current_start / frame_seqlen}")
            #     print(f"kv_cache['global_end_index'] = {kv_cache['global_end_index']}")
            #     print(f"kv_cache['local_end_index'] = {kv_cache['local_end_index']}")
            #     print(f"num_new_tokens = {num_new_tokens}")

            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                if selected_index is not None:
                    num_evicted_tokens = selected_index.shape[0] + kv_cache["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                else:
                    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"need roll")
                #     print(f"num_rolled_tokens: {num_rolled_tokens / frame_seqlen}")
                #     print(f"num_evicted_tokens: {num_evicted_tokens / frame_seqlen}")
                #     print(f"sink_tokens: {sink_tokens / frame_seqlen}")

                # Compute updated local indices
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                if selected_index is not None:
                    act_local_end_index = local_end_index - roped_key.shape[1] + selected_index.shape[0]
                    act_local_start_index = act_local_end_index - selected_index.shape[0]
                    act_current_end = current_end - roped_key.shape[1] + selected_index.shape[0]

                # Construct full k, v for attention computation (without modifying the original cache)
                # Create temporary k, v for computation
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                
                # Apply rolling update to the temporary cache
                temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert new key/value into the temporary cache
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": sink_tokens,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": roped_key[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }
                if act_local_start_index is not None:
                    cache_update_info["local_end_index"]=act_local_end_index
                    cache_update_info["local_start_index"]=act_local_start_index
                    cache_update_info["current_end"]=act_current_end
                    current_end=act_current_end

                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"used kv cache size: local_end_index - local_start_index = {local_end_index - local_start_index}")
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                if selected_index is not None:
                    act_local_end_index = local_end_index - roped_key.shape[1] + selected_index.shape[0]
                    act_local_start_index = act_local_end_index - selected_index.shape[0]
                    act_current_end = current_end - roped_key.shape[1] + selected_index.shape[0]

                # Construct full k, v for attention computation (without modifying the original cache)
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    # if selected_index is not None:
                    #     print(1)
                    # if not sparsity_params["frame_idx"] in self.k_cache:
                    #     self.k_cache[sparsity_params["frame_idx"]] = {}  
                    #     self.v_cache[sparsity_params["frame_idx"]] = {}  
                    # self.k_cache[sparsity_params["frame_idx"]][sparsity_params["denoising_step"]] = roped_key[:, roped_offset:roped_offset + write_len]
                    # self.v_cache[sparsity_params["frame_idx"]][sparsity_params["denoising_step"]] = v[:, roped_offset:roped_offset + write_len]
                    if sparsity_params is not None and sparsity_params.get("denoising_step", 0) == 0:
                        self.k_cache[0] = roped_key[:, roped_offset:roped_offset + write_len]
                        self.v_cache[0] = v[:, roped_offset:roped_offset + write_len]

                    if sparsity_params is not None and selected_index is not None and sparsity_params.get("denoising_step", 0) > 0:
                        frame_len = frame_seqlen * grid_sizes[0][0].item()
                        local_end_index = local_start_index + frame_len
                        current_end = local_end_index
                        # act_local_start_index = None
                        temp_k[:, write_start_index:local_end_index][:, ~selected_index] = temp_k[:, write_start_index - frame_len:local_end_index - frame_len][:, ~selected_index]
                        temp_v[:, write_start_index:local_end_index][:, ~selected_index] = temp_v[:, write_start_index - frame_len:local_end_index - frame_len][:, ~selected_index]
                        temp_k[:, write_start_index:local_end_index][:, selected_index] = roped_key[:, roped_offset:roped_offset + write_len]
                        temp_v[:, write_start_index:local_end_index][:, selected_index] = v[:, roped_offset:roped_offset + write_len]
                        write_len = local_end_index - local_start_index
                        roped_key = temp_k[:, write_start_index:local_end_index]
                        v = temp_v[:, write_start_index:local_end_index]
                    else:
                        temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                        temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": roped_key[:, roped_offset:roped_offset + write_len],
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }
                if act_local_start_index is not None:
                    cache_update_info["local_end_index"]=act_local_end_index
                    cache_update_info["local_start_index"]=act_local_start_index
                    cache_update_info["current_end"]=act_current_end
                    current_end=act_current_end
                # if cache_update_info["local_end_index"]%4680!=0 or cache_update_info["current_end"]%4680!=0 or cache_update_info["local_start_index"]%4680!=0:
                #     print("error")

            # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            #     print(f"local_start_index: {local_start_index}, local_end_index: {local_end_index}")

            # Use temporary k, v to compute attention
            block_frames_size = roped_query.shape[1]
            st=time.time()
            if sink_tokens > 0:
                # Concatenate sink tokens and local window tokens, keeping total length strictly below max_attention_size
                local_budget = self.max_attention_size - sink_tokens
                k_sink = temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]
                # if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
                #     print(f"local_budget: {local_budget}")
                if local_budget > 0:
                    local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                    k_local = temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v[:, local_start_for_window:local_end_index]
                    k_cat = torch.cat([k_sink, k_local], dim=1)
                    v_cat = torch.cat([v_sink, v_local], dim=1)
                else:
                    k_cat = k_sink
                    v_cat = v_sink
                attn_hist_end = max(0, k_cat.shape[1] - block_frames_size)
                attn_block_size = block_frames_size
                is_already_cache = attn_hist_end > 0
                broadcast_self = False
                if use_pab and is_already_cache and frame_idx is not None:
                    current_count = self.pab_self_count.get(frame_idx, 0)
                    broadcast_self, next_count = pab_manager.if_broadcast_self(denoising_step, current_count)
                    self.pab_self_count[frame_idx] = next_count
                    if broadcast_self and frame_idx not in self.pab_last_self:
                        broadcast_self = False
                if sparsity_params is not None and sparsity_params.get("is_sparsity", False) and sparsity_params.get("denoising_step", 0) in sparsity_params.get("sparsity_steps", []) and is_already_cache:
                    st=time.time()
                    profile_core = _profile_enabled(sparsity_params, roped_query)
                    if sparsity_params.get("skip_reuse_merge", False):
                        if self.head_mask[sparsity_params["denoising_step"]].sum()==0:
                            if profile_core:
                                profile_start, profile_end = _event_pair()
                                profile_start.record()
                            current_attn_output,current_logsumexp = attention(
                                roped_query,
                                k_cat[:,attn_hist_end:],
                                v_cat[:,attn_hist_end:],
                                return_hist=True,
                            )
                            if profile_core:
                                profile_end.record()
                                _profile_add(sparsity_params, "cur_attn_ms", _event_elapsed(profile_start, profile_end))
                        else:
                            head_mask_=self.head_mask[sparsity_params["denoising_step"]]
                            if profile_core:
                                profile_start, profile_end = _event_pair()
                                profile_start.record()
                            current_attn_output,current_logsumexp, x_hist, softmax_lse_hist = attention(
                                roped_query,
                                k_cat,
                                v_cat,
                                return_hist=True,block_size=attn_block_size,head_mask=head_mask_
                            )
                            if profile_core:
                                profile_end.record()
                                _profile_add(sparsity_params, "cur_attn_ms", _event_elapsed(profile_start, profile_end))
                            self.attn_output_past[:,:,head_mask_] = x_hist[:,:,head_mask_]
                            self.logsumexp[head_mask_] = softmax_lse_hist[head_mask_]
                        x = current_attn_output.to(x)
                    else:
                        head_mask_=self.head_mask[sparsity_params["denoising_step"]]
                        if profile_core:
                            profile_start, profile_end = _event_pair()
                            profile_start.record()
                        current_attn_output,current_logsumexp, self.attn_output_past, self.logsumexp = attention(
                            roped_query,
                            k_cat,
                            v_cat,
                            return_hist=True,
                            block_size=attn_block_size,
                            head_mask=head_mask_,
                            attn_output_past=self.attn_output_past,
                            logsumexp=self.logsumexp,
                        )
                        if profile_core:
                            profile_end.record()
                            _profile_add(sparsity_params, "cur_attn_ms", _event_elapsed(profile_start, profile_end))
                        x = current_attn_output.to(x)
                    if use_pab and frame_idx is not None:
                        self.pab_last_self[frame_idx] = x
                else:
                    if broadcast_self:
                        x = self.pab_last_self[frame_idx]
                    elif sparsity_params is not None and sparsity_params.get("is_sparsity", False):
                        st=time.time()
                        profile_core = _profile_enabled(sparsity_params, roped_query)
                        if profile_core:
                            profile_start, profile_end = _event_pair()
                            profile_start.record()
                        x, softmax_lse, x_hist, softmax_lse_hist = attention(
                            roped_query,
                            k_cat,
                            v_cat,
                            block_size=attn_block_size,
                            return_hist=True
                        )
                        if profile_core:
                            profile_end.record()
                            _profile_add(sparsity_params, "full_attn_ms", _event_elapsed(profile_start, profile_end))
                        self.attn_output_past = x_hist
                        self.logsumexp = softmax_lse_hist
                        self.softmax_lse = softmax_lse
                        self.x=x
                        if use_pab and frame_idx is not None:
                            self.pab_last_self[frame_idx] = x
                    else:
                        use_svg2 = sparsity_params.get("use_svg2", False) if sparsity_params is not None else False
                        if use_svg2:
                            if sparsity_params["denoising_step"] == 0 and roped_query.shape[1]==k_cat.shape[1]:
                                self.processor.centroids_init = False
                            x = self.processor.attention_core_logic(roped_query.transpose(1,2),
                                                                  k_cat.transpose(1,2),v_cat.transpose(1,2),t[0],sparsity_params=sparsity_params).transpose(1,2)
                        elif sparsity_params is not None and sparsity_params.get("sparge_attn",0)>0:
                            if spas_sage2_attn_meansim_topk_cuda is None:
                                raise ImportError("spas_sage_attn is unavailable, but --sparge_attn was requested")
                            x = spas_sage2_attn_meansim_topk_cuda(roped_query.transpose(1,2), k_cat.transpose(1,2), v_cat.transpose(1,2), topk=sparsity_params.get("sparge_attn",0), is_causal=False).transpose(1,2)
                        else:
                            sttt=time.time()
                            # segmented_pooled_attention(roped_query,k_cat,grid_sizes,16,3)
                            profile_core = _profile_enabled(sparsity_params, roped_query)
                            if profile_core:
                                profile_start, profile_end = _event_pair()
                                profile_start.record()
                            x = attention(
                                roped_query,
                                k_cat,
                                v_cat,
                            )
                            if profile_core:
                                profile_end.record()
                                _profile_add(sparsity_params, "full_attn_ms", _event_elapsed(profile_start, profile_end))
                        if use_pab and frame_idx is not None:
                            self.pab_last_self[frame_idx] = x
                            # if k_cat.shape[1]//1560==12:
                            #     # segmented_pooled_attention(roped_query,k_cat,grid_sizes,16,3)
                            #     x=mix_linear_attention(roped_query,k_cat,v_cat,grid_sizes,16,1560)
                            #     # x = mix_softmax_flash_attention(roped_query,k_cat,v_cat,grid_sizes,4,1560,attention)
                            # if sparsity_params is not None and sparsity_params["denoising_step"]==0:
                            #     self.cache_score[sparsity_params["frame_idx"]]=compute_attention_prob(roped_query[:,:roped_query.shape[1]//3],k_cat[:,:roped_query.shape[1]//3]).cpu()
                if sparsity_params is not None and sparsity_params.get("is_calibrate", False) and is_already_cache:
                    temp,temp_logsumexp = attention(
                      roped_query,
                      k_cat[:, :attn_hist_end],
                      v_cat[:, :attn_hist_end],
                      block_size=0,
                      return_hist=True
                    )
                    temp_in,temp_logsumexp_in = attention(
                      roped_query,
                      k_cat[:, attn_hist_end:],
                      v_cat[:, attn_hist_end:],
                      block_size=0,
                      return_hist=True
                    )
                    # full_temp,full_temp_logsumexp = attention(
                    #    roped_query,
                    #    kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                    #    kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                    #    block_size=block_size,
                    #    return_hist=True
                    # )
                    if sparsity_params.get("denoising_step", 0) in sparsity_params.get("sparsity_steps", []):
                        step_index=sparsity_params["denoising_step"]
                        cos=torch.nn.functional.cosine_similarity(temp.float(),self.attn_output_past.float(),dim=-1).mean((0, 1))
                        cos_in=torch.nn.functional.cosine_similarity(temp_in.float(),self.attn_in_past.float(),dim=-1).mean((0, 1))
                        if hasattr(self, "logsumexp"):
                            lse_delta=(temp_logsumexp.float() - self.logsumexp.float()).abs().mean(dim=-1)
                        else:
                            lse_delta=torch.full_like(cos, float("inf"))
                        self.coses[step_index] = cos
                        self.coses_in[step_index] = cos_in
                        self.lse_deltas[step_index] = lse_delta
                        if step_index not in self.cos_sums:
                            self.cos_sums[step_index] = cos.detach().clone()
                            self.cos_counts[step_index] = 1
                        else:
                            self.cos_sums[step_index] = self.cos_sums[step_index] + cos.detach()
                            self.cos_counts[step_index] += 1
                        if step_index not in self.lse_delta_sums:
                            self.lse_delta_sums[step_index] = lse_delta.detach().clone()
                            self.lse_delta_counts[step_index] = 1
                        else:
                            self.lse_delta_sums[step_index] = self.lse_delta_sums[step_index] + lse_delta.detach()
                            self.lse_delta_counts[step_index] += 1
                        if step_index not in self.cos_in_sums:
                            self.cos_in_sums[step_index] = cos_in.detach().clone()
                            self.cos_in_counts[step_index] = 1
                        else:
                            self.cos_in_sums[step_index] = self.cos_in_sums[step_index] + cos_in.detach()
                            self.cos_in_counts[step_index] += 1
                        head_sim_th = sparsity_params.get("head_sim_th_by_step", {}).get(step_index, sparsity_params["head_sim_th"])
                        head_lse_th = sparsity_params.get("head_lse_th", 0.05)
                        self.head_mask[step_index] = (cos < head_sim_th) | (lse_delta > head_lse_th)
                        # if sparsity_params["denoising_step"]-1 in self.head_mask:
                        last_mask=self.head_mask[sparsity_params["denoising_step"]]
                        self.attn_output_past[:,:,last_mask]=temp[:,:,last_mask]
                        self.logsumexp[last_mask]=temp_logsumexp[last_mask]
                    else:
                        self.attn_output_past=temp
                        self.attn_in_past=temp_in
                        self.logsumexp=temp_logsumexp
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                profile_core = _profile_enabled(sparsity_params, roped_query)
                if profile_core:
                    profile_start, profile_end = _event_pair()
                    profile_start.record()
                x = attention(
                    roped_query,
                    temp_k[:, window_start:local_end_index],
                    temp_v[:, window_start:local_end_index]
                )
                if profile_core:
                    profile_end.record()
                    _profile_add(sparsity_params, "full_attn_ms", _event_elapsed(profile_start, profile_end))
            att_time=time.time()-st

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info), att_time
        else:
            return x, att_time

    def reset_pab_state(self):
        self.pab_self_count.clear()
        self.pab_last_self.clear()
class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 layer_i=0):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps, layer_i)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.block_idx = layer_i
        self.pab_cross_count = {}
        self.pab_last_cross = {}

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        sparsity_params=None,
        t=None,
        selected_index=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames = e.shape[1]
        frame_seqlen = (grid_sizes[0, 1] * grid_sizes[0, 2]).item()
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        if selected_index is not None:
            e = (self.modulation.unsqueeze(1) + e).unsqueeze(2).expand(-1,-1,frame_seqlen,-1,-1).flatten(1,2)[:,selected_index].chunk(6, dim=2)
        else:
            e = (self.modulation.unsqueeze(1) + e).unsqueeze(2).expand(-1,-1,frame_seqlen,-1,-1).flatten(1,2).chunk(6, dim=2)
        # e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        self_attn_input = (self.norm1(x) * (1 + e[1][:,:,0]) + e[0][:,:,0])
        if self_attn_input.is_cuda:
            self_attn_start = torch.cuda.Event(enable_timing=True)
            self_attn_end = torch.cuda.Event(enable_timing=True)
            self_attn_start.record()
            self_attn_result = self.self_attn(
                self_attn_input,
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch, sparsity_params, t, selected_index)
            self_attn_end.record()
            self_attn_end.synchronize()
            att_time = self_attn_start.elapsed_time(self_attn_end) / 1000.0
        else:
            self_attn_st = time.perf_counter()
            self_attn_result = self.self_attn(
                self_attn_input,
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch, sparsity_params, t, selected_index)
            att_time = time.perf_counter() - self_attn_st

        if kv_cache is not None:
            y, cache_update_info, _raw_self_attn_time = self_attn_result
        else:
            y, _raw_self_attn_time = self_attn_result
            cache_update_info = None

        # with amp.autocast(dtype=torch.float32):
        x = x + (y * e[2][:,:,0])

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            frame_idx = None if sparsity_params is None else sparsity_params.get("frame_idx")
            denoising_step = None if sparsity_params is None else sparsity_params.get("denoising_step")
            use_pab = bool(PABManager is not None and sparsity_params is not None and sparsity_params.get("use_pab", False))
            cross_residual = None
            if use_pab and frame_idx is not None:
                pab_manager = PABManager(
                    PABConfig(
                        cross_broadcast=True,
                        cross_threshold=sparsity_params.get("pab_cross_threshold"),
                        cross_range=sparsity_params.get("pab_cross_range", 1),
                        cross_reuse_steps=sparsity_params.get("pab_cross_reuse_steps"),
                    )
                )
                current_count = self.pab_cross_count.get(frame_idx, 0)
                broadcast_cross, next_count = pab_manager.if_broadcast_cross(denoising_step, current_count)
                self.pab_cross_count[frame_idx] = next_count
                if broadcast_cross:
                    cross_residual = self.pab_last_cross.get(frame_idx)

            if cross_residual is None:
                cross_residual = self.cross_attn(
                    self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache
                )
                if use_pab and frame_idx is not None:
                    self.pab_last_cross[frame_idx] = cross_residual

            x = x + cross_residual
            y = self.ffn(
                (self.norm2(x) * (1 + e[4][:,:,0]) + e[3][:,:,0])
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5][:,:,0]
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        
        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info, att_time
        else:
            return x, att_time

    def reset_pab_state(self):
        self.self_attn.reset_pab_state()
        self.pab_cross_count.clear()
        self.pab_last_cross.clear()


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e, selected_index=None, frame_seqlen=1560):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames = e.shape[1]
        if selected_index is not None:
            e = (self.modulation.unsqueeze(1) + e).unsqueeze(2).expand(-1,-1,frame_seqlen,-1,-1).flatten(1,2)[:,selected_index].chunk(2, dim=2)
        else:
            e = (self.modulation.unsqueeze(1) + e).unsqueeze(2).expand(-1,-1,frame_seqlen,-1,-1).flatten(1,2).chunk(2, dim=2)
        x = (self.head(self.norm(x) * (1 + e[1][:,:,0]) + e[0][:,:,0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.x_cache={0:{},-1:{}}

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps,layer_i)
            for layer_i in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def reset_pab_state(self):
        for block in self.blocks:
            if hasattr(block, "reset_pab_state"):
                block.reset_pab_state()

    def _set_gradient_checkpointing(self, module=None, value=False, **kwargs):
        if 'enable' in kwargs:
            value = kwargs['enable']
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if (not dist.is_initialized() or dist.get_rank() == 0) and DEBUG:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # # debug
        # DEBUG = False
        # if DEBUG:
        #     num_frames = 9
        #     frame_seqlen = 256

        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            pass

        # import imageio
        # import numpy as np
        # from torch.nn.attention.flex_attention import create_mask

        # mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
        #                    padded_length, KV_LEN=total_length + padded_length, device=device)
        # import cv2
        # mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
        # imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Perform the rolling operation
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                    
                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
            
            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                if current_end%4680!=0:
                    print("error")
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False,
        sparsity_params=None,
        selected_index=None
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # print(f"x.device: {x[0].device}, t.device: {t.device}, context.device: {context.device}, seq_len: {seq_len}")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print("patch embedding done")
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """
        st=time.time()

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # print("time embedding done")
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        # print("text embedding done")
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            sink_recache_after_switch=sink_recache_after_switch
        )
        # print("kwargs done")
        def create_custom_forward(module):
            def custom_forward(*args_in, **kwargs_in):
                return module(*args_in, **kwargs_in)
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        att_time_all = 0
        if selected_index is not None:
            x=x[:,selected_index]
        for block_index, block in enumerate(self.blocks):
            # print(f"block_index: {block_index}")
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "sparsity_params": sparsity_params,
                        "t": t,
                        "selected_index": selected_index
                    }
                )
                # print(f"forward checkpointing")
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info, att_time = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                    att_time_all += att_time
                else:
                    x, att_time = result
                    att_time_all += att_time
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "sparsity_params": sparsity_params,
                        "t": t,
                        "selected_index":selected_index
                    }
                )
                # print(f"forward no checkpointing")
                result = block(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info, att_time = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x,att_time = result
                att_time_all+=att_time
        # log_gpu_memory(f"in _forward_inference: {x[0].device}")
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        # if sparsity_params["denoising_step"] in [0,3]:
        #     self.x=x
        # else:
        #     x=self.x
        # if not sparsity_params["frame_idx"] in self.x_cache:
        #     self.x_cache[sparsity_params["frame_idx"]] = {}
        use_selected_index = bool(sparsity_params is not None and sparsity_params.get("use_selected_index", False))
        if use_selected_index and sparsity_params.get("frame_idx", 0)>1 and sparsity_params.get("denoising_step", 0)==0 and selected_index is None:
            self.x_cache[-1]=self.x_cache[0]
        # if sparsity_params["frame_idx"]>1 and sparsity_params["denoising_step"]>0 and sparsity_params["denoising_step"] in []:
        #     cos_sim=torch.nn.functional.cosine_similarity(self.x_cache[-1][sparsity_params["denoising_step"]],x,dim=-1)
        #     x[cos_sim>0.9]=self.x_cache[-1][sparsity_params["denoising_step"]][cos_sim>0.9]
        # if sparsity_params["frame_idx"]>0:
        frame_seqlen = (grid_sizes[0][1] * grid_sizes[0][2]).item()
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2), selected_index=selected_index, frame_seqlen=frame_seqlen)
        if use_selected_index:
            if selected_index is None:
                self.x_cache[0][sparsity_params.get("denoising_step", 0)]=x[:,:,-(grid_sizes[0][1]*grid_sizes[0][2]):].reshape(x.shape[0],-1,x.shape[-1]).detach().clone()
            else:
                recover_x = self.x_cache[0][sparsity_params.get("denoising_step", 0)].clone()
                recover_x[:,selected_index]=x.reshape(x.shape[0],-1,x.shape[-1])
                x=recover_x
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        torch.cuda.synchronize()
        # print(time.time()-st)
        return torch.stack(x),att_time_all

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        pass
        raise NotImplementedError()
    
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        frame_seqlen = (grid_sizes[0][1] * grid_sizes[0][2]).item()
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2), frame_seqlen=frame_seqlen)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
