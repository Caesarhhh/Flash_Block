import torch
from torch import nn
import os
import json
import triton
import triton.language as tl
import time
from jetengine_ext.utils.context import get_context
from jetengine_ext.engine.sequence import RunType
from jetengine_ext.kernels.triton.attention import sparse_attn_varlen
# from jetengine.kernels.triton.attention import fused_kv_cache_attention
# from jetengine.kernels.triton.attention import fused_kv_cache_attention_v5
from flash_attn import flash_attn_with_kvcache, flash_attn_func, flash_attn_varlen_func
from copy import deepcopy

_ATTN_CONTEXT_DUMP_COUNT = 0
_PROFILE_ATTENTION = os.environ.get("TRADO_SYNC_PROFILE", "0") == "1"
_SKIP_PYTHON_CACHE_WRITEBACK = os.environ.get(
    "TRADO_SKIP_PY_CACHE_WRITEBACK", "1"
).lower() not in ("0", "false", "off")
_ATTN_CONTEXT_DUMP_PATH = os.environ.get("TRADO_ATTN_CONTEXT_DUMP")
_ATTN_CONTEXT_DUMP_MIN_CONTEXT = int(os.environ.get("TRADO_ATTN_CONTEXT_DUMP_MIN_CONTEXT", "0"))
_ATTN_CONTEXT_DUMP_LIMIT = int(os.environ.get("TRADO_ATTN_CONTEXT_DUMP_LIMIT", "256"))
_ATTN_CONTEXT_DUMP_VALUES = os.environ.get("TRADO_ATTN_CONTEXT_DUMP_VALUES", "0") == "1"


def _maybe_dump_attention_context(context, q, sparsity_params):
    global _ATTN_CONTEXT_DUMP_COUNT
    dump_path = _ATTN_CONTEXT_DUMP_PATH
    if not dump_path or context.run_type == RunType.PREFILL:
        return
    min_context = _ATTN_CONTEXT_DUMP_MIN_CONTEXT
    if min_context and context.context_lens is not None and int(context.context_lens.detach().sum().item()) < min_context:
        return
    limit = _ATTN_CONTEXT_DUMP_LIMIT
    if _ATTN_CONTEXT_DUMP_COUNT >= limit:
        return
    _ATTN_CONTEXT_DUMP_COUNT += 1

    def tensor_stats(x):
        if x is None:
            return None
        y = x.detach()
        if y.numel() == 0:
            return {"shape": list(y.shape), "numel": 0}
        y_cpu = y.to("cpu")
        return {
            "shape": list(y_cpu.shape),
            "numel": int(y_cpu.numel()),
            "min": int(y_cpu.min().item()),
            "max": int(y_cpu.max().item()),
            "mean": float(y_cpu.float().mean().item()),
            "sum": int(y_cpu.long().sum().item()),
            "zero_count": int((y_cpu == 0).sum().item()),
        }

    sp0 = sparsity_params[0] if sparsity_params is not None and len(sparsity_params) and sparsity_params[0] is not None else None
    payload = {
        "idx": _ATTN_CONTEXT_DUMP_COUNT - 1,
        "pid": os.getpid(),
        "device": str(q.device),
        "run_type": str(context.run_type),
        "q_shape_flat": list(q.shape),
        "block_length": int(context.block_length),
        "batch_size_from_q": int(q.shape[0] // context.block_length) if context.block_length else 0,
        "context_lens": tensor_stats(context.context_lens),
        "delta_cached_lens": tensor_stats(context.delta_cached_lens),
        "block_tables_shape": list(context.block_tables.shape) if context.block_tables is not None else None,
        "need_update_mask_shape": list(context.need_update_kvcache_mask.shape) if context.need_update_kvcache_mask is not None else None,
        "need_update_mask_true": int(context.need_update_kvcache_mask.sum().item()) if context.need_update_kvcache_mask is not None else None,
        "dirty_mask_shape": list(context.dirty_mask.shape) if context.dirty_mask is not None else None,
        "dirty_mask_true": int(context.dirty_mask.sum().item()) if context.dirty_mask is not None else None,
        "has_sparsity": sp0 is not None,
    }
    if _ATTN_CONTEXT_DUMP_VALUES:
        payload["context_lens_values"] = (
            context.context_lens.detach().to("cpu").tolist()
            if context.context_lens is not None
            else None
        )
        payload["delta_cached_lens_values"] = (
            context.delta_cached_lens.detach().to("cpu").tolist()
            if context.delta_cached_lens is not None
            else None
        )
    if sp0 is not None:
        for key in ("need_update_kvcache_idx", "no_need_update_kvcache_idx"):
            value = sp0.get(key, [])
            payload[f"{key}_len"] = int(value.numel() if torch.is_tensor(value) else len(value))
    os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
    with open(dump_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    assert 1==0
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache_python(key, value, k_cache, v_cache, slot_mapping):
    """
    对应 store_kvcache_kernel 的 Python 等价实现。
    参数:
        key_flat, value_flat: [N, D]  (D = num_kv_heads * head_dim)
        k_cache, v_cache: [num_blocks, block_size, num_kv_heads, head_dim]
        slot_mapping: [N]
    """
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 展平为 [N, D] 方便操作
    key_flat   = key.reshape(N, D)
    value_flat = value.reshape(N, D)
    num_blocks, block_size, num_kv_heads, head_dim = k_cache.shape
    assert D == num_kv_heads * head_dim

    # reshape key/value
    key_view = key_flat.view(N, num_kv_heads, head_dim)
    value_view = value_flat.view(N, num_kv_heads, head_dim)

    for i in range(N):
        slot = slot_mapping[i].item()
        block_id = slot // block_size
        offset_in_block = slot % block_size

        # 写入到正确的 block 和位置
        k_cache[block_id, offset_in_block, :, :] = key_view[i]
        v_cache[block_id, offset_in_block, :, :] = value_view[i]



def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
    # store_kvcache_python(key, value, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.attn_output_current = None
        self.attn_output_past = None
        self.logsumexp = None
        self.o = None
        self.cache_ids = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        pass
    
import torch

def make_varlen_inputs_from_batch(q, k=None, v=None):
    """
    支持 Q 的 head 数与 KV 不同的情况（MQA / GQA）。
    自动生成 flash_attn_varlen_func 所需输入参数。

    Args:
        q: [B, Lq, Hq, D]
        k: [B, Lk, Hkv, D] (Hkv 可以小于 Hq)
        v: [B, Lk, Hkv, D]

    Returns:
        q_var, k_var, v_var,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k
    """
    device = q.device
    B, Lq, Hq, D = q.shape

    if k is None:
        k = q
    if v is None:
        v = k
    Lk, Hkv = k.shape[1], k.shape[2]

    # Flatten batch + sequence → varlen 格式
    q_var = q.reshape(B * Lq, Hq, D).to(device)
    k_var = k.reshape(B * Lk, Hkv, D).to(device)
    v_var = v.reshape(B * Lk, Hkv, D).to(device)

    # Cumulative sequence lengths
    cu_seqlens_q = torch.arange(0, (B + 1) * Lq, Lq, dtype=torch.int32, device=device)
    cu_seqlens_k = torch.arange(0, (B + 1) * Lk, Lk, dtype=torch.int32, device=device)

    max_seqlen_q = Lq
    max_seqlen_k = Lk

    return q_var, k_var, v_var, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k

def zero_dirty_tokens(attn_output_past, dirty_tokens=None, mask=None):
    """
    高效清零指定的 head 通道 (第二维)，无 Python for 循环。
    attn_output_past: [B, H, N, D]
    dirty_tokens: list[list[int]]，每个 batch 的 dirty head 索引
    """
    B, H, N, D = attn_output_past.shape
    # 构造 mask，True 表示该 head 需要清零
    if mask is None:
        mask = torch.zeros((B, H), dtype=torch.bool, device=attn_output_past.device)

        # 用 advanced indexing 一次性填充 True
        # 将 dirty_tokens 展平成索引对
        batch_idx = torch.arange(B, device=attn_output_past.device)
        all_b = []
        all_h = []
        for b, dirty in enumerate(dirty_tokens):
            if dirty:
                all_b.extend([b] * len(dirty))
                all_h.extend(dirty)
        if all_b:  # 避免空情况
            mask[torch.tensor(all_b, device=attn_output_past.device),
                 torch.tensor(all_h, device=attn_output_past.device)] = True
    
    # 扩展 mask 形状为 [B, H, 1, 1] 以便广播
    attn_output_past.masked_fill_(mask, 0)
    return mask

class BlockAttention(Attention):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__(num_heads, head_dim, scale, num_kv_heads)
        self.not_update=0

    def fake_sparsity(self,q,context,need_update_kvcache_idx,no_need_update_kvcache_idx,need_update_kvcache_mask,past_attn_output,past_logsumexp,logsumexp,attn_output_past_copy,logsumexp_copy,o_copy):
        batch_size=q.shape[0]

        if len(need_update_kvcache_idx)>0:
            # attn_sel = past_attn_output.index_select(0, need_update_kvcache_idx)
            # logsum_sel = past_logsumexp.index_select(0, need_update_kvcache_idx)
            # self.attn_output_past.index_copy_(0, need_update_kvcache_idx, attn_sel)
            # self.logsumexp.index_copy_(0, need_update_kvcache_idx, logsum_sel)
            
            attn_output_past_copy[:batch_size] = torch.where(
                need_update_kvcache_mask.permute(0,2,1,3),
                past_attn_output,
                attn_output_past_copy[:batch_size]
            )
            logsumexp_copy[:batch_size] = torch.where(
                need_update_kvcache_mask.permute(0,2,1,3),
                past_logsumexp,
                logsumexp_copy[:batch_size]
            )
        if len(no_need_update_kvcache_idx)>0:
            # current_logsumexp = logsumexp.index_select(0,no_need_update_kvcache_idx)
            # current_attn_output = o.index_select(0,no_need_update_kvcache_idx)
            # past_logsumexp = self.logsumexp.index_select(0,no_need_update_kvcache_idx)
            current_logsumexp=logsumexp
            current_attn_output=o_copy
            past_logsumexp=logsumexp_copy[:batch_size]
            #current_logsumexp=logsumexp
            #current_attn_output=o
            #past_logsumexp=self.logsumexp[:batch_size]
            m = torch.maximum(past_logsumexp, current_logsumexp)       
            exp_past, exp_cur = (past_logsumexp-m).exp(), (current_logsumexp-m).exp()  # 解包
            denom = exp_past + exp_cur
            attn_output_past_copy[:batch_size][no_need_update_kvcache_idx]=attn_output_past_copy[:batch_size][no_need_update_kvcache_idx].masked_fill_(context.dirty_mask[no_need_update_kvcache_idx].permute(0,2,1,3), 0)
            veri_attn_output = (
                (attn_output_past_copy[:batch_size] * exp_past + current_attn_output * exp_cur) / denom
            )
            o_copy=torch.where(need_update_kvcache_mask.permute(0,2,1,3), o_copy, veri_attn_output).contiguous()

        return o_copy,attn_output_past_copy[:batch_size],logsumexp_copy[:batch_size]

    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sparsity_params={}):
        o: torch.Tensor
        profile_attention = _PROFILE_ATTENTION
        profile_start_event = None
        profile_end_event = None
        if profile_attention and q.is_cuda:
            profile_start_event = torch.cuda.Event(enable_timing=True)
            profile_end_event = torch.cuda.Event(enable_timing=True)
            profile_start_event.record()
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        batch_size=q.shape[0] // context.block_length
        _maybe_dump_attention_context(context, q, sparsity_params)
        has_sparsity = (
            context.run_type != RunType.PREFILL
            and sparsity_params is not None
            and sparsity_params[0] is not None
        )

        need_update_cache = True

        # if sparsity_params is not None and sparsity_params[0] is not None and len(sparsity_params[0]["dirty_tokens"])>0 and len(sparsity_params[0]["dirty_tokens"]) < sparsity_params[0]["sparsity_ratio"]:
        #     need_update_cache = False

        #if sparsity_params is not None and sparsity_params[0] is not None and self.not_update >= sparsity_params[0]["sparsity_ratio"]-1:
        #    need_update_cache = False
        #    self.not_update += 1

        # if need_update_cache:
        k_cache, v_cache = self.k_cache, self.v_cache
        # else:
        #     k_cache, v_cache = None, None
            
        should_store_whole = (context.run_type == RunType.PREFILL)
        if should_store_whole and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

            
        if context.run_type == RunType.PREFILL:
            o = sparse_attn_varlen(q, k, v,
                                cu_seqlens_q=context.cu_seqlens_q,
                                cu_seqlens_k=context.cu_seqlens_k,
                                staircase_size=context.block_length)
        else:
            q = q.view(-1, context.block_length, self.num_heads, self.head_dim)
            k = k.view(-1, context.block_length, self.num_kv_heads, self.head_dim)
            v = v.view(-1, context.block_length, self.num_kv_heads, self.head_dim)

            st=time.time()

            # if need_update_cache:
            if has_sparsity:
                need_update_kvcache_mask = context.need_update_kvcache_mask
                # dirty_mask = context.need_update_kvcache_mask.to(q)
                # o, logsumexp,past_attn_output ,past_logsumexp  = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                #                            cache_seqlens=context.context_lens,
                #                            block_table=context.block_tables,
                #                            causal=False, return_softmax_lse=True, block_size=q.shape[1])  # Assuming non-causal for benchmark consistency
                # logsumexp = logsumexp.to(o)
                # o_fake,past_attn_output_fake,logsumexp_fake=self.fake_sparsity(q,context,need_update_kvcache_idx,no_need_update_kvcache_idx,need_update_kvcache_mask,past_attn_output,past_logsumexp,logsumexp,deepcopy(self.attn_output_past),deepcopy(self.logsumexp),deepcopy(o))
                
                need_store_history_mask = context.need_store_history_mask
                use_all_update_no_store_fastpath = context.all_update_no_store_fastpath
                if use_all_update_no_store_fastpath:
                    o, _ = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                                                cache_seqlens=context.context_lens,
                                                block_table=context.block_tables,
                                                causal=False, return_softmax_lse=True, block_size=0)
                else:
                    all_update_store_only = context.all_update_store_only
                    cache_updated_in_kernel = (
                        _SKIP_PYTHON_CACHE_WRITEBACK
                        and self.attn_output_past.dtype == torch.float32
                        and need_store_history_mask is not None
                    )
                    o, _, _past_attn_output, _past_logsumexp, self_attn_output_past, self_logsumexp = flash_attn_with_kvcache(
                                                q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                                                cache_seqlens=context.context_lens, delta_cache_seqlens=context.delta_cached_lens,
                                                block_table=context.block_tables,
                                                causal=False, return_softmax_lse=True, block_size=q.shape[1], num_splits=1,
                                                attn_output_past=self.attn_output_past[:batch_size],
                                                logsumexp=self.logsumexp[:batch_size],
                                                need_update_kvcache_mask=None if all_update_store_only else need_update_kvcache_mask,
                                                need_store_history_mask=need_store_history_mask,
                                                dirty_mask=None if all_update_store_only else context.dirty_mask)  # Assuming non-causal for benchmark consistency

                    if not cache_updated_in_kernel:
                        if self_attn_output_past.dtype != self.attn_output_past.dtype:
                            self_attn_output_past = self_attn_output_past.to(self.attn_output_past.dtype)
                        if need_store_history_mask is None:
                            self.attn_output_past[:batch_size] = self_attn_output_past
                            self.logsumexp[:batch_size] = self_logsumexp
                        else:
                            store_mask_bqhd = need_store_history_mask.permute(0, 2, 1, 3)
                            self.attn_output_past[:batch_size] = torch.where(
                                store_mask_bqhd,
                                self_attn_output_past,
                                self.attn_output_past[:batch_size],
                            )
                            self.logsumexp[:batch_size] = torch.where(
                                store_mask_bqhd,
                                self_logsumexp,
                                self.logsumexp[:batch_size],
                            )
                # past_attn_output, past_logsumexp = o, logsumexp
                # context_lens=deepcopy(context.context_lens)
                # past_attn_output,past_logsumexp = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=None, v=None,
                #                             cache_seqlens=context.context_lens,
                #                             block_table=context.block_tables,
                #                             causal=False, return_softmax_lse=True)
                #o, logsumexp = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                #                            cache_seqlens=context.context_lens,
                #                            block_table=context.block_tables,
                #                            causal=False, return_softmax_lse=True)
                
                # past_logsumexp = past_logsumexp.to(past_attn_output)
                # logsumexp = logsumexp.to(o)
                # o, logsumexp,past_attn_output,past_logsumexp = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                #                             cache_seqlens=context.context_lens,
                #                             block_table=context.block_tables,
                #                             causal=False, return_softmax_lse=True, block_size=q.shape[1])  # Assuming non-causal for benchmark consistency
            else:
                o, _ = flash_attn_with_kvcache(q, k_cache=k_cache, v_cache=v_cache, k=k, v=v,
                                            cache_seqlens=context.context_lens,
                                            block_table=context.block_tables,
                                            causal=False, return_softmax_lse=True, block_size=0)  # Assuming non-causal for benchmark consistency
            # else:
            #     #q_var, k_var, v_var, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = make_varlen_inputs_from_batch(q, k, v)
            #     #o, logsumexp, _ = flash_attn_varlen_func(q_var, k=k_var, v=v_var, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            #     #                                         max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
            #     #                                causal=False, return_attn_probs=True)  # Assuming non-causal for benchmark consistency
            #     o, logsumexp, _ = flash_attn_func(q, k=k, v=v, causal=False, return_attn_probs=True)
                # logsumexp=logsumexp[None,:]
                #print(f"use kv attn {time.time()-st}")
            if False and sparsity_params is not None and sparsity_params[0] is not None :

                if len(need_update_kvcache_idx)>0:
                    # attn_sel = past_attn_output.index_select(0, need_update_kvcache_idx)
                    # logsum_sel = past_logsumexp.index_select(0, need_update_kvcache_idx)
                    # self.attn_output_past.index_copy_(0, need_update_kvcache_idx, attn_sel)
                    # self.logsumexp.index_copy_(0, need_update_kvcache_idx, logsum_sel)
                    
                    self.attn_output_past[:batch_size] = torch.where(
                        need_update_kvcache_mask.permute(0,2,1,3),
                        past_attn_output,
                        self.attn_output_past[:batch_size]
                    )

                    self.logsumexp[:batch_size] = torch.where(
                        need_update_kvcache_mask.permute(0,2,1,3),
                        past_logsumexp,
                        self.logsumexp[:batch_size]
                    )

                if len(no_need_update_kvcache_idx)>0:
                    # current_logsumexp = logsumexp.index_select(0,no_need_update_kvcache_idx)
                    # current_attn_output = o.index_select(0,no_need_update_kvcache_idx)
                    # past_logsumexp = self.logsumexp.index_select(0,no_need_update_kvcache_idx)

                    current_logsumexp=logsumexp
                    current_attn_output=o
                    past_logsumexp=self.logsumexp[:batch_size]
                    #current_logsumexp=logsumexp
                    #current_attn_output=o
                    #past_logsumexp=self.logsumexp[:batch_size]

                    m = torch.maximum(past_logsumexp, current_logsumexp)
                    exp_past, exp_cur = (past_logsumexp-m).exp(), (current_logsumexp-m).exp()  # 解包
                    exp_past[:batch_size][no_need_update_kvcache_idx]=exp_past[:batch_size][no_need_update_kvcache_idx].masked_fill_(context.dirty_mask[no_need_update_kvcache_idx].permute(0,2,1,3).to(q.device), 0)
                    denom = exp_past + exp_cur
                    # self.attn_output_past[:batch_size][no_need_update_kvcache_idx]=self.attn_output_past[:batch_size][no_need_update_kvcache_idx].masked_fill_(context.dirty_mask[no_need_update_kvcache_idx].permute(0,2,1,3).to(q.device), 0)
                    veri_attn_output = (
                        (self.attn_output_past[:batch_size] * exp_past + current_attn_output * exp_cur) / denom
                    ).to(o)
                    o=torch.where(need_update_kvcache_mask.permute(0,2,1,3), o, veri_attn_output).contiguous()
                    if o.isnan().sum()>0 or o.isinf().sum()>0:
                        print("o is nan or inf")

                    # o.index_copy_(0,no_need_update_kvcache_idx,veri_attn_output.index_select(0,no_need_update_kvcache_idx))
                    
            
        if profile_start_event is not None:
            profile_end_event.record()
            context.profile_attention_events.append((profile_start_event, profile_end_event))

        o = o.view(-1, self.num_heads * self.head_dim)
        return o

        
