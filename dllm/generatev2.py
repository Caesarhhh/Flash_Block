# adpated from SADR https://github.com/JetAstra/SDAR/blob/main/generate.py

import argparse
import json
import os
import torch
import time
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import random
from typing import Dict, Any, Optional, Tuple


def _append_jsonl(path, payload):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

class DynamicCacheWithPadOverwrite(DynamicCache):
    """
    DynamicCache 扩展版：
    - update_with_pad_overwrite(): 生成阶段按 padding 覆盖；
    - merge_with_cache_for_inference(): 推理阶段根据 attention_mask 定位新增 token 覆盖；
      packed multi-block 训练阶段显式 append 到 prefix cache 后面。
      完全并行，无 for 循环。
    """

    def __init__(self, pad_id: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = pad_id

    # ==========================================================
    # ✅ 写入（生成阶段）不变
    # ==========================================================
    def update_with_pad_overwrite(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return key_states, value_states

        old_k = self.key_cache[layer_idx]
        old_v = self.value_cache[layer_idx]

        if not old_k.numel():
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
            return key_states, value_states

        B, H, L_old, D = old_k.shape
        _, _, L_new, _ = key_states.shape

        if L_new > L_old:
            pad_k = old_k.new_zeros(B, H, L_new - L_old, D)
            pad_v = old_v.new_zeros(B, H, L_new - L_old, D)
            old_k = torch.cat([old_k, pad_k], dim=2)
            old_v = torch.cat([old_v, pad_v], dim=2)

        overwrite_len = min(L_new, L_old)
        old_k[:, :, :overwrite_len, :] = key_states[:, :, :overwrite_len, :]
        old_v[:, :, :overwrite_len, :] = value_states[:, :, :overwrite_len, :]

        if L_new > L_old:
            old_k[:, :, L_old:L_new, :] = key_states[:, :, L_old:L_new, :]
            old_v[:, :, L_old:L_new, :] = value_states[:, :, L_old:L_new, :]

        self.key_cache[layer_idx] = old_k
        self.value_cache[layer_idx] = old_v
        return old_k, old_v

    # ==========================================================
    # ✅ 推理阶段并行合并 (完全符合你的逻辑)
    # ==========================================================
    def merge_with_cache_for_inference(
        self,
        key_states: torch.Tensor,     # [B, H, L_new, D]
        value_states: torch.Tensor,   # [B, H, L_new, D]
        layer_idx: int,
        attention_mask: torch.Tensor, # [B, L_total], 已扩展
        append_to_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对每个样本：
        - 根据 attention_mask 找到新 token 的位置（最后 L_new 个 1）；
        - 如果这些位置超出 cache 长度则扩展；
        - 否则直接覆盖。
        packed multi-block 训练里，当前 tokens 是额外 append 的 probe blocks，
        此时直接返回 [prefix_cache, current_probe_kv]。
        """
        if len(self.key_cache) <= layer_idx:
            return key_states, value_states

        past_k = self.key_cache[layer_idx]
        past_v = self.value_cache[layer_idx]

        if not past_k.numel():
            return key_states, value_states

        B, H, L_cache, D = past_k.shape
        _, _, L_new, _ = key_states.shape
        device = past_k.device

        if append_to_cache:
            return (
                torch.cat([past_k, key_states], dim=-2),
                torch.cat([past_v, value_states], dim=-2),
            )

        # 1️⃣ 每个样本的有效长度（最后一个 1 的下标 + 1）
        valid_len = attention_mask.sum(dim=-1)  # [B]
        start_pos = torch.clamp(valid_len - L_new, min=0)  # [B]
        end_pos = valid_len  # [B]

        # 2️⃣ 判断是否需要扩展 cache
        max_valid = valid_len.max().item()
        if max_valid > L_cache:
            pad_len = max_valid - L_cache
            pad_k = past_k.new_zeros(B, H, pad_len, D)
            pad_v = past_v.new_zeros(B, H, pad_len, D)
            past_k = torch.cat([past_k, pad_k], dim=2)
            past_v = torch.cat([past_v, pad_v], dim=2)
            L_cache = max_valid

        # 3️⃣ 构建 idx，确定哪些位置要被覆盖
        idx = torch.arange(L_cache, device=device).unsqueeze(0)  # [1, L_cache]
        cover_mask = (idx >= start_pos.unsqueeze(1)) & (idx < end_pos.unsqueeze(1))  # [B, L_cache]

        # 4️⃣ 直接用 mask 覆盖 (broadcast)
        cover_mask_4d = cover_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L_cache, 1]
        merged_k = past_k.clone()
        merged_v = past_v.clone()

        # 右对齐新 token
        merged_k[cover_mask_4d.expand_as(merged_k)] = key_states[:, :, :L_new, :].reshape(-1)
        merged_v[cover_mask_4d.expand_as(merged_v)] = value_states[:, :, :L_new, :].reshape(-1)

        return merged_k, merged_v



def chunked_prefill(
    model,
    input_ids: torch.Tensor,
    past_key_values,
    chunk_size: int = 512,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs
):
    """
    分块 prefill 一个序列，以支持更长的长度而不触发 OOM。
    
    参数:
        model: CausalLM 模型
        input_ids: [B, L] 的输入张量
        past_key_values: 满足 transformers Cache 接口的对象
        chunk_size: 每个 chunk 的长度
        attention_mask: [B, 1, L, L] 或 [B, L, L] 的全量注意力掩码
        position_ids: [B, L] 的全量位置 ID
    """
    B, L = input_ids.shape
    device = input_ids.device
    
    for i in range(0, L, chunk_size):
        end_idx = min(i + chunk_size, L)
        chunk_input_ids = input_ids[:, i:end_idx]
        
        # 1. 构造当前 chunk 的 mask
        # Query 范围是 [i:end_idx], Key 范围是 [0:end_idx]
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                curr_mask = attention_mask[:, :, i:end_idx, :end_idx]
            else:
                curr_mask = attention_mask[:, i:end_idx, :end_idx]
        else:
            curr_mask = None
            
        # 2. 构造当前 chunk 的 position_ids
        if position_ids is not None:
            curr_pos = position_ids[:, i:end_idx]
        else:
            curr_pos = torch.arange(i, end_idx, device=device).unsqueeze(0).expand(B, -1)
            
        # 3. 前向传播并更新 Cache
        model(
            chunk_input_ids,
            attention_mask=curr_mask,
            past_key_values=past_key_values,
            position_ids=curr_pos,
            use_cache=True,
            store_kv=True,
            chunk_prefill=True, # 告诉模型内部使用 append 模式更新 cache
            **kwargs
        )

def top_k_logits(logits, k):
    if k <= 0:
        return logits
    else:
        values, _ = torch.topk(logits, k)
        min_values = values[..., -1, None]
        return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)



def top_p_logits(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_mask = cumulative_probs > p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False
    mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool),
                                 -1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask_indices, float('-inf'))
    return logits


def sample_with_temperature_topk_topp(logits, temperature=1.0, top_k=0, top_p=1.0):
    orig_shape = logits.shape[:-1]    # [batch, block]
    vocab_size = logits.shape[-1]

    logits = logits.reshape(-1, vocab_size)  # [batch*block, vocab]

    if temperature != 1.0:
        logits = logits / temperature
    if top_k > 0:
        logits = top_k_logits(logits, top_k)
    if top_p < 1.0:
        logits = top_p_logits(logits, top_p)
    probs = F.softmax(logits, dim=-1)  # shape: [batch*block, vocab]
    assert probs.dim() == 2
    token = torch.multinomial(probs, num_samples=1)  # [batch*block, 1]
    token_prob = torch.gather(probs, -1, token)     # [batch*block, 1]

    return token.view(*orig_shape), token_prob.view(*orig_shape)


def get_num_transfer_tokens(block_length, steps):
    base = block_length // steps
    remainder = block_length % steps
    num_transfer_tokens = torch.zeros(steps, dtype=torch.int64) + base
    num_transfer_tokens[:remainder] += 1
    return num_transfer_tokens

import sys

_last_lines = 0  # 记录上次打印的行数

def live_print(text: str):
    """
    在终端中动态刷新多行输出（覆盖旧内容）

    参数:
        text (str): 要显示的多行字符串
    """
    global _last_lines

    # 上移光标到上次输出的起始位置
    if _last_lines:
        sys.stdout.write(f"\033[{_last_lines}A")  # 光标上移
        sys.stdout.write("\033[J")  # 清除光标以下的所有内容

    # 打印新的内容
    print(text, flush=True)

    # 记录当前输出的行数
    _last_lines = text.count('\n') + 1

def sample_and_select_transfer_tokens(
    logits,
    temperature,
    top_k,
    top_p,
    cur_x,
    mask_index,
    num_transfer_tokens,
    step,
    remasking_strategy="sequential",
    confidence_threshold=0.5,
    sample_with_temperature_topk_topp=None,
):
    """
    对 logits 进行采样，并根据 remasking_strategy 生成 transfer_index 掩码。

    返回:
        x0, x0_p, transfer_index
    """

    # Use a self-consistent greedy teacher signal for training.
    # Previously x0 came from argmax while x0_p came from sampled tokens, which
    # made confidence-based transfer decisions depend on a different token.
    probs = F.softmax(logits, dim=-1)
    x0 = logits.argmax(-1)
    x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if torch.is_tensor(num_transfer_tokens):
        transfer_budget = int(num_transfer_tokens[step].item())
    elif isinstance(num_transfer_tokens, (list, tuple)):
        transfer_budget = int(num_transfer_tokens[step])
    else:
        transfer_budget = int(num_transfer_tokens)
    transfer_budget = max(0, transfer_budget)

    # Initialize transfer_index
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)

    # Strategy: sequential
    if remasking_strategy == 'sequential':
        for j in range(cur_x.shape[0]):
            if transfer_budget == 0:
                continue
            if mask_index[j].any():
                first_mask_index = mask_index[j].nonzero(as_tuple=True)[0].min().item()
                transfer_index[j, first_mask_index:first_mask_index + transfer_budget] = True
            else:
                raise ValueError(f"No mask tokens found in current block for batch {j}.")

    # Strategy: low_confidence_static
    elif remasking_strategy == 'low_confidence_static':
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
        for j in range(confidence.shape[0]):
            if transfer_budget == 0:
                continue
            _, idx = torch.topk(confidence[j], transfer_budget)
            transfer_index[j, idx] = True

    # Strategy: low_confidence_dynamic
    elif remasking_strategy == 'low_confidence_dynamic':
        confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
        for j in range(confidence.shape[0]):
            high_conf_mask = confidence[j] > confidence_threshold
            num_high_confidence = high_conf_mask.sum()
            if transfer_budget == 0:
                continue
            if num_high_confidence==0:
                _, idx = torch.topk(confidence[j], 1)
                transfer_index[j, idx] = True
            elif num_high_confidence <= transfer_budget:
                transfer_index[j] = high_conf_mask
            else:
                _, idx = torch.topk(confidence[j], transfer_budget)
                transfer_index[j, idx] = True

    else:
        raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")
    return x0, x0_p, transfer_index


def sample_and_select_transfer_tokens_packed(
    logits,
    temperature,
    top_k,
    top_p,
    cur_x,
    mask_index,
    num_transfer_tokens,
    step,
    block_length,
    remasking_strategy="sequential",
    confidence_threshold=0.5,
    sample_with_temperature_topk_topp=None,
):
    """
    Packed multi-block version of sample_and_select_transfer_tokens.

    cur_x is [B, num_blocks * block_length]. Each block independently transfers
    num_transfer_tokens[step] tokens, matching per-block DLLM denoising.
    """
    probs = F.softmax(logits, dim=-1)
    x0 = logits.argmax(-1)
    x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

    if torch.is_tensor(num_transfer_tokens):
        transfer_budget = int(num_transfer_tokens[step].item())
    elif isinstance(num_transfer_tokens, (list, tuple)):
        transfer_budget = int(num_transfer_tokens[step])
    else:
        transfer_budget = int(num_transfer_tokens)
    transfer_budget = max(0, transfer_budget)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    if transfer_budget == 0:
        return x0, x0_p, transfer_index

    B, query_len = cur_x.shape
    if query_len % block_length != 0:
        raise ValueError(
            f"Packed query length {query_len} is not divisible by block_length {block_length}."
        )

    confidence = torch.where(mask_index, x0_p, torch.full_like(x0_p, -torch.inf))
    for block_start in range(0, query_len, block_length):
        block_end = block_start + block_length
        block_mask = mask_index[:, block_start:block_end]
        block_confidence = confidence[:, block_start:block_end]

        for b in range(B):
            if not block_mask[b].any():
                continue

            if remasking_strategy == "sequential":
                first_mask_index = block_mask[b].nonzero(as_tuple=True)[0].min().item()
                budget = min(transfer_budget, int(block_mask[b].sum().item()))
                transfer_index[
                    b, block_start + first_mask_index:block_start + first_mask_index + budget
                ] = True

            elif remasking_strategy == "low_confidence_static":
                budget = min(transfer_budget, int(block_mask[b].sum().item()))
                _, idx = torch.topk(block_confidence[b], budget)
                transfer_index[b, block_start + idx] = True

            elif remasking_strategy == "low_confidence_dynamic":
                high_conf_mask = block_confidence[b] > confidence_threshold
                num_high_confidence = int(high_conf_mask.sum().item())
                if num_high_confidence == 0:
                    _, idx = torch.topk(block_confidence[b], 1)
                    transfer_index[b, block_start + idx] = True
                elif num_high_confidence <= transfer_budget:
                    transfer_index[b, block_start:block_end] = high_conf_mask
                else:
                    _, idx = torch.topk(block_confidence[b], transfer_budget)
                    transfer_index[b, block_start + idx] = True

            else:
                raise ValueError(f"Unknown remasking strategy: {remasking_strategy}")

    return x0, x0_p, transfer_index


def _select_multi_block_probe_starts(
    input_ids: torch.Tensor,
    padding_id: int,
    block_length: int,
    num_probe_blocks: int,
    min_prefix_blocks: int,
    random_select: bool = True,
):
    valid_lens = (input_ids != padding_id).sum(dim=1)
    min_valid_len = int(valid_lens.min().item())

    first_start = max(1, int(min_prefix_blocks)) * block_length
    last_start = ((min_valid_len - block_length) // block_length) * block_length
    if last_start < first_start:
        return []

    starts = list(range(first_start, last_start + 1, block_length))
    if num_probe_blocks > 0 and len(starts) > num_probe_blocks:
        if random_select:
            probe_indices = torch.randperm(
                len(starts), device=input_ids.device
            )[:num_probe_blocks]
            probe_indices = torch.sort(probe_indices).values.tolist()
        else:
            probe_indices = torch.linspace(
                0,
                len(starts) - 1,
                steps=num_probe_blocks,
                device=input_ids.device,
            ).round().long().tolist()
        starts = [starts[i] for i in probe_indices]
    return starts


def multi_block_probe_teacher_student(
    teacher_model,
    model,
    input_ids: torch.Tensor,
    mask_id: int = 151669,
    padding_id: int = 151643,
    block_length: int = 8,
    denoising_steps: int = 8,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.85,
    tokenizer=None,
    train_sparsity_ratio: int = 100,
    sparse_loss_weight: float = 0.5,
    dense_loss_weight: float = 0.5,
    num_probe_blocks: int = 0,
    min_prefix_blocks: int = 1,
    random_probe_blocks: bool = True,
):
    """
    将多个 mask block 同时作为 query 输入。

    真实 token prefix 只 prefill 一次；每个 mask block 的 4D attention mask
    只允许看它对应的真实 prefix 和自己的 mask block，不允许看后续真实 token
    或其它 probe 的 mask block。
    """
    starts = _select_multi_block_probe_starts(
        input_ids=input_ids,
        padding_id=padding_id,
        block_length=block_length,
        num_probe_blocks=int(num_probe_blocks),
        min_prefix_blocks=int(min_prefix_blocks),
        random_select=random_probe_blocks,
    )
    if not starts:
        _, teacher_logits, student_logits, kl_loss, acc, acc_dense, kl_loss_sparse, kl_loss_dense = block_teacher_student_from_cache(
            teacher_model,
            model,
            input_ids,
            None,
            0,
            block_length=block_length,
            valid_len=0,
            mask_id=mask_id,
            padding_id=padding_id,
            tokenizer=tokenizer,
            denoising_steps=denoising_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            remasking_strategy=remasking_strategy,
            confidence_threshold=confidence_threshold,
            train_sparsity_ratio=train_sparsity_ratio,
            sparse_loss_weight=sparse_loss_weight,
            dense_loss_weight=dense_loss_weight,
        )
        return teacher_logits, student_logits, kl_loss, acc, acc_dense, kl_loss_sparse, kl_loss_dense

    teacher_model.eval()
    device = input_ids.device
    B = input_ids.shape[0]
    num_probe_blocks = len(starts)
    query_len = num_probe_blocks * block_length
    max_prefix_len = max(starts)
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps).to(device)

    prefix = input_ids[:, :max_prefix_len].contiguous()
    prefix_attn_mask = build_dllm_attention_mask(
        prefix, padding_id, block_length, None, extra_tokens=0
    )

    query_x = torch.full(
        (B, query_len), mask_id, dtype=input_ids.dtype, device=device
    )
    position_ids = torch.cat(
        [
            torch.arange(start, start + block_length, device=device)
            for start in starts
        ],
        dim=0,
    ).unsqueeze(0).expand(B, -1)

    attention_mask = torch.zeros(
        (B, 1, query_len, max_prefix_len + query_len),
        dtype=torch.float32,
        device=device,
    )
    past_attention_mask = torch.zeros_like(attention_mask)
    current_attention_mask = torch.zeros(
        (B, 1, query_len, query_len),
        dtype=torch.float32,
        device=device,
    )
    for probe_idx, block_start in enumerate(starts):
        q_start = probe_idx * block_length
        q_end = q_start + block_length
        k_start = max_prefix_len + q_start
        k_end = k_start + block_length
        past_attention_mask[:, :, q_start:q_end, :block_start] = 1.0
        current_attention_mask[:, :, q_start:q_end, q_start:q_end] = 1.0
        attention_mask[:, :, q_start:q_end, :block_start] = 1.0
        attention_mask[:, :, q_start:q_end, k_start:k_end] = 1.0

    # ===== Teacher: dense two-step denoising over all probe blocks =====
    teacher_cache = DynamicCacheWithPadOverwrite(pad_id=padding_id)
    with torch.no_grad():
        chunked_prefill(
            teacher_model,
            prefix,
            chunk_size=512,
            attention_mask=prefix_attn_mask,
            past_key_values=teacher_cache,
        )

    teacher_x = query_x.clone()
    teacher_logits = None
    with torch.no_grad():
        for step in range(0, 2):
            mask_index = teacher_x == mask_id
            if mask_index.sum() == 0:
                break
            logits = teacher_model(
                teacher_x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=teacher_cache,
                use_cache=True,
                store_kv=False,
                sparsity_params=None,
            ).logits
            x0, _, transfer_index = sample_and_select_transfer_tokens_packed(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cur_x=teacher_x,
                mask_index=mask_index,
                num_transfer_tokens=num_transfer_tokens,
                step=step,
                block_length=block_length,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                sample_with_temperature_topk_topp=sample_with_temperature_topk_topp,
            )
            teacher_x[transfer_index & mask_index] = x0[transfer_index & mask_index]
            if teacher_logits is None:
                teacher_logits = torch.zeros_like(logits)
            teacher_logits[mask_index] = logits[mask_index]
            if step == 0:
                first_transfer_index = transfer_index & mask_index
            if step == 1:
                second_mask_index = mask_index

    # ===== Student: step 0 stores prefix/current split stats for packed fusion =====
    student_cache = DynamicCacheWithPadOverwrite(pad_id=padding_id)
    with torch.no_grad():
        chunked_prefill(
            model,
            prefix,
            chunk_size=512,
            attention_mask=prefix_attn_mask,
            past_key_values=student_cache,
        )

    student_x = query_x.clone()
    sparsity_params = {
        "step": 0,
        "dirty_tokens": [[] for _ in range(B)],
        "sparsity_ratio": 999,
        "packed_probe": True,
        "past_attention_mask": past_attention_mask,
        "current_attention_mask": current_attention_mask,
    }
    model(
        student_x,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=student_cache,
        use_cache=True,
        store_kv=False,
        sparsity_params=sparsity_params,
    )
    student_x[first_transfer_index] = teacher_x[first_transfer_index]

    dirty_tokens = [
        transfer_index_.nonzero(as_tuple=True)[0].tolist()
        for transfer_index_ in first_transfer_index
    ]
    sparsity_params = {
        "step": 1,
        "dirty_tokens": dirty_tokens,
        "sparsity_ratio": 999,
        "packed_probe": True,
        "force_no_update_cache": True,
        "current_attention_mask": current_attention_mask,
    }
    outputs_student = model(
        student_x,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=student_cache,
        use_cache=True,
        store_kv=False,
        sparsity_params=sparsity_params,
    )
    student_logits = outputs_student.logits

    T = 1.0
    teacher_log_probs = F.log_softmax(teacher_logits[second_mask_index].float() / T, dim=-1)
    student_log_probs = F.log_softmax(student_logits[second_mask_index].float() / T, dim=-1)
    kl_loss = F.kl_div(
        student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True
    ) * (T * T)
    acc = (
        teacher_logits[second_mask_index].argmax(-1)
        == student_logits[second_mask_index].argmax(-1)
    ).float().mean().item()

    outputs_student = model(
        student_x,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=student_cache,
        use_cache=True,
        store_kv=False,
        sparsity_params=None,
    )
    student_logits_dense = outputs_student.logits
    student_log_probs_dense = F.log_softmax(
        student_logits_dense[second_mask_index].float() / T, dim=-1
    )
    kl_loss_dense = F.kl_div(
        student_log_probs_dense, teacher_log_probs, reduction="batchmean", log_target=True
    ) * (T * T)
    acc_dense = (
        teacher_logits[second_mask_index].argmax(-1)
        == student_logits_dense[second_mask_index].argmax(-1)
    ).float().mean().item()

    kl_loss_agg = kl_loss * sparse_loss_weight + kl_loss_dense * dense_loss_weight
    dump_path = os.environ.get("TRADO_TRAIN_REUSE_DUMP")
    if dump_path:
        with torch.no_grad():
            first_counts = first_transfer_index.sum(dim=1).detach().cpu().tolist()
            second_counts = second_mask_index.sum(dim=1).detach().cpu().tolist()
            dirty_lens = [len(tokens) for tokens in dirty_tokens]
            teacher_argmax = teacher_logits[second_mask_index].argmax(-1)
            sparse_argmax = student_logits[second_mask_index].argmax(-1)
            dense_argmax = student_logits_dense[second_mask_index].argmax(-1)
            _append_jsonl(dump_path, {
                "source": "train",
                "path": "multi_block_probe",
                "batch_size": int(B),
                "block_length": int(block_length),
                "denoising_steps": int(denoising_steps),
                "confidence_threshold": float(confidence_threshold),
                "train_sparsity_ratio": int(train_sparsity_ratio),
                "kernel_sparsity_ratio": 999,
                "num_probe_blocks": int(num_probe_blocks),
                "query_len": int(query_len),
                "max_prefix_len": int(max_prefix_len),
                "probe_starts": [int(start) for start in starts[:64]],
                "first_transfer_counts": [int(x) for x in first_counts],
                "second_mask_counts": [int(x) for x in second_counts],
                "dirty_lens": [int(x) for x in dirty_lens],
                "dirty_examples": [list(map(int, tokens[:16])) for tokens in dirty_tokens[:8]],
                "dirty_len_mean": float(sum(dirty_lens) / len(dirty_lens)) if dirty_lens else 0.0,
                "second_mask_total": int(second_mask_index.sum().item()),
                "kl_sparse": float(kl_loss.detach().float().item()),
                "kl_dense": float(kl_loss_dense.detach().float().item()),
                "acc_sparse": float(acc),
                "acc_dense": float(acc_dense),
                "sparse_teacher_match": float((teacher_argmax == sparse_argmax).float().mean().item()),
                "dense_teacher_match": float((teacher_argmax == dense_argmax).float().mean().item()),
            })
    return (
        teacher_logits,
        student_logits,
        kl_loss_agg,
        acc,
        acc_dense,
        kl_loss.item(),
        kl_loss_dense.item(),
    )


def block_prefill_teacher_student(
    teacher_model,
    model,
    input_ids: torch.Tensor,
    mask_id: int = 151669,
    padding_id: int = 151643,
    block_length: int = 8,
    denoising_steps: int = 8,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.85,
    tokenizer = None,
    prompt_lengths: Optional[torch.Tensor] = None,
    train_sparsity_ratio: int = 100,
    sparse_loss_weight: float = 0.5,
    dense_loss_weight: float = 0.5,
    multi_block_probe: bool = False,
    multi_block_probe_num: int = 0,
    multi_block_min_prefix_blocks: int = 1,
    multi_block_random_probe: bool = True,
):
    """
    支持输入含 padding，block 拼在 padding 后。
    逻辑：
      1. prefill 整个 prompt（跳过 padding token）
      2. teacher dense denoising
      3. student (sparsity=1) forward 一次生成 token
      4. student (sparsity=100) 再 forward 并计算 KL loss
    """
    if multi_block_probe:
        teacher_logits, student_logits, kl_loss, acc, acc_dense, kl_loss_sparse, kl_loss_dense = multi_block_probe_teacher_student(
            teacher_model,
            model,
            input_ids=input_ids,
            mask_id=mask_id,
            padding_id=padding_id,
            block_length=block_length,
            denoising_steps=denoising_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            remasking_strategy=remasking_strategy,
            confidence_threshold=confidence_threshold,
            tokenizer=tokenizer,
            train_sparsity_ratio=train_sparsity_ratio,
            sparse_loss_weight=sparse_loss_weight,
            dense_loss_weight=dense_loss_weight,
            num_probe_blocks=multi_block_probe_num,
            min_prefix_blocks=multi_block_min_prefix_blocks,
            random_probe_blocks=multi_block_random_probe,
        )
    else:
        # x, past_key_values, cur_context_len, valid_len = block_inference_prefix(
        #     teacher_model, input_ids, num_infer_blocks=random.randint(1,2),block_length=block_length,tokenizer=tokenizer
        # )
        x, teacher_logits, student_logits, kl_loss,acc, acc_dense, kl_loss_sparse, kl_loss_dense = block_teacher_student_from_cache(
        teacher_model,model, input_ids, None, 0, block_length=block_length, valid_len=0,tokenizer=tokenizer, prompt_lengths=prompt_lengths,
        denoising_steps=denoising_steps,
        remasking_strategy=remasking_strategy,
        confidence_threshold=confidence_threshold,
        train_sparsity_ratio=train_sparsity_ratio,
        sparse_loss_weight=sparse_loss_weight,
        dense_loss_weight=dense_loss_weight,
        )
    return teacher_logits, student_logits, kl_loss,acc, acc_dense, kl_loss_sparse, kl_loss_dense

def build_dllm_attention_mask(input_ids, padding_id, block_size, prompt_lengths, extra_tokens=0):
    """
    构造 DLLM 注意力 mask:
      - 纯 Block Causal Mask: 所有 token 按照 block_size 划分
      - Block内全注意，Block间因果注意（只能看前面块）
      - padding token 不参与 attention (mask=0)
    """
    device = input_ids.device
    B, seq_len = input_ids.shape
    
    # prompt_lengths is ignored as per user request
    # "prompt_lengths 根本屁用没有，所有token都是block causal的"
    
    # 初始化为全 0 (不可见)
    attn_mask = torch.zeros((B, 1, seq_len, seq_len), dtype=torch.float32, device=device)
    
    num_blocks = (seq_len + block_size - 1) // block_size
    
    for i in range(B):
        # 1. 纯 Block Causal 逻辑
        for b in range(num_blocks):
            block_start = b * block_size
            block_end = min(block_start + block_size, seq_len)
            
            # 当前块可以看到:
            # 1. 自己 (block_start:block_end) -> (block_start:block_end)
            # 2. 之前所有块 (0:block_start)
            # 合并: block_start:block_end 可以看到 0:block_end
            attn_mask[i, :, block_start:block_end, :block_end] = 1.0

        # 2. Padding Mask
        # 如果 block 中包含了 pad，是否需要 mask 掉？
        # "padding token 不参与 attention"
        # 意思是被 pad 的列应该是 0
        pad_indices = (input_ids[i] == padding_id).nonzero(as_tuple=True)[0]
        if len(pad_indices) > 0:
            attn_mask[i, :, :, pad_indices] = 0.0

    return attn_mask

import torch

def move_pad_to_end(x: torch.Tensor, pad_id: int):
    """
    高并行版本：将输入 tensor 中的 pad token (== pad_id) 全部移到序列末尾，
    并返回新的 attention_mask。

    参数:
        x: [batch, seq_len] 或 [batch, seq_len, dim] 的 Tensor
        pad_id: int，要移动到末尾的 pad 标记 id

    返回:
        new_x: 同形状 Tensor，pad 全部移到序列末尾
        attention_mask: [batch, seq_len]，1 表示有效 token，0 表示 pad
    """
    if x.ndim == 2:
        valid_mask = (x != pad_id).int()  # [B, L]
    else:
        valid_mask = (x[..., 0] != pad_id).int()  # [B, L]

    # 按照有效 token 排序（有效的在前，pad 在后）
    sort_keys = -valid_mask  # 有效 token 变大（-1），pad 变小（0）
    sorted_indices = torch.argsort(sort_keys, dim=1, stable=True)  # [B, L]

    # gather 重新排列
    gather_indices = sorted_indices.unsqueeze(-1).expand_as(x)
    new_x = torch.gather(x, dim=1, index=gather_indices)

    # 生成新的 attention mask：有效 token = 1, pad = 0
    new_attention_mask = torch.gather(valid_mask, dim=1, index=sorted_indices)

    return new_x, new_attention_mask

def prepare_mask_for_generation(
    x: torch.Tensor, pad_id: int, mask_id: int, block_size: int
):
    """
    将每个样本中最靠前的 block_size 个 pad token 替换为 mask；
    若 pad 数量不足 block_size，则在末尾补 pad 后再替换。
    返回:
      - new_x: 替换并补齐后的序列
      - attention_mask: 1=非pad
      - replace_mask: bool mask, 被替换的位置 (B, L_total)
      - replace_indices: 被替换 pad 的索引 (B, block_size)
    """
    B, L = x.shape
    device, dtype = x.device, x.dtype

    # ---- Step 1. pad 掩码与计数 ----
    pad_mask = (x == pad_id)
    pad_counts = pad_mask.sum(dim=1)

    # ---- Step 2. 若 pad 不足 block_size，末尾补 pad ----
    need_pad = torch.clamp(block_size - pad_counts, min=0)
    max_extra = need_pad.max().item()
    if max_extra > 0:
        pad_extra = torch.full((B, max_extra), pad_id, dtype=dtype, device=device)
        x = torch.cat([x, pad_extra], dim=1)
        pad_mask = (x == pad_id)
        pad_counts = pad_mask.sum(dim=1)
        L = x.size(1)

    # ---- Step 3. 获取 pad 的真实位置索引 ----
    idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    pad_indices = torch.where(pad_mask, idx, torch.full_like(idx, L))

    # 每个样本的 pad 索引排序（pad 都在后面，但这里保证顺序稳定）
    pad_indices_sorted, _ = torch.sort(pad_indices, dim=1)

    # 取前 block_size 个 pad 位置（真实 pad，不够的样本现在已经补齐）
    replace_indices = pad_indices_sorted[:, :block_size]  # [B, block_size]

    # ---- Step 4. 构造替换 mask 并替换 ----
    replace_mask = torch.zeros_like(x, dtype=torch.bool)
    replace_mask.scatter_(1, replace_indices, True)
    new_x = torch.where(replace_mask, torch.full_like(x, mask_id), x)

    # ---- Step 5. attention mask ----
    attention_mask = (new_x != pad_id).int()

    return new_x, attention_mask, replace_mask, replace_indices

@torch.no_grad()
def block_inference_prefix(
    model,
    input_ids: torch.Tensor,
    block_length: int,
    num_infer_blocks: int,
    denoising_steps: int = 8,
    mask_id: int = 151669,
    padding_id: int = 151643,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.85,
    tokenizer = None
):
    """
    DLLM 版本 prefix 推理:
      - 全连接 attention；
      - padding 屏蔽；
      - KV cache 不污染；
      - 推理 num_infer_blocks 个 block。
    """
    device = input_ids.device
    B, L = input_ids.shape
    past_key_values = DynamicCacheWithPadOverwrite(pad_id=padding_id)

    # 这里的 input_ids 本身就是 prompt
    # 这里的 input_ids 本身就是 prompt
    prompt_lengths = (input_ids != padding_id).sum(dim=1)
    attn_mask = build_dllm_attention_mask(input_ids, padding_id, block_length, None, extra_tokens=0)
    valid_len = (input_ids != padding_id).sum(dim=1)

    # ===== prefill =====
    model(
        input_ids,
        attention_mask=attn_mask,
        past_key_values=past_key_values,
        use_cache=True,
        store_kv=True,
    )

    # ===== 推理若干个 block =====
    cur_context_len = L
    x = input_ids
    for _ in range(num_infer_blocks):
        x, attention_mask, replace_mask, replace_indices=prepare_mask_for_generation(x,padding_id,mask_id,block_length)
        cur_x = x[replace_mask].reshape(B,-1)

        for step in range(denoising_steps + 1):
            mask_index = (cur_x == mask_id)
            if mask_index.sum() == 0:
                model(cur_x,
                      attention_mask=attention_mask,
                      position_ids=replace_indices,
                      past_key_values=past_key_values,
                      use_cache=True,
                      store_kv=True)
                break
            logits = model(
                cur_x,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
                sparsity_params=None,
                position_ids=replace_indices
            ).logits

            x0, _, transfer_index = sample_and_select_transfer_tokens(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cur_x=cur_x,
                mask_index=mask_index,
                num_transfer_tokens=block_length,
                step=step,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                sample_with_temperature_topk_topp=sample_with_temperature_topk_topp,
            )
            cur_x[transfer_index*mask_index] = x0[transfer_index*mask_index]

        x[replace_mask] = cur_x.flatten()
        cur_context_len += block_length

    return x, past_key_values, cur_context_len, valid_len

def build_dirty_attention_mask(
    attention_mask,  # [B, K]
    q_len,           # Q = 4
    dirty_lists      # list[list[int]]
):
    B, K = attention_mask.shape
    Q = q_len

    # [B, Q, K]
    attn_mask = attention_mask[:, None, :].expand(B, Q, K).clone()

    block_start = attention_mask.sum(dim=1) - Q
    block_start = block_start.long()

    for b in range(B):
        kv_end = block_start[b]
        for q_idx in dirty_lists[b]:
            attn_mask[b, q_idx, :kv_end] = 0

    return attn_mask  # [B, Q, K]


def block_teacher_student_from_cache(
    teacher_model,
    model,
    x: torch.Tensor,
    past_key_values,
    cur_context_len: int,
    valid_len: torch.Tensor,
    block_length: int,
    denoising_steps: int = 8,
    mask_id: int = 151669,
    padding_id: int = 151643,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    remasking_strategy: str = "low_confidence_dynamic",
    confidence_threshold: float = 0.85,
    tokenizer=None, sparsity=False,
    prompt_lengths: Optional[torch.Tensor] = None,
    train_sparsity_ratio: int = 100,
    sparse_loss_weight: float = 0.5,
    dense_loss_weight: float = 0.5,
):
    """
    从 prefix 缓存继续推理一个 block，并执行 teacher-student KL loss。
    """
    teacher_model.eval()
    device = x.device
    B, total_len = x.shape
    past_key_values = DynamicCacheWithPadOverwrite(pad_id=padding_id)
    # 如果没传 prompt_lengths，则假设 x 除去 padding 都是 prompt (首次调用)
    # 如果没传 prompt_lengths，则假设 x 除去 padding 都是 prompt (首次调用)
    # if prompt_lengths is None:
    #     prompt_lengths = (x != padding_id).sum(dim=1)
        
    attn_mask = build_dllm_attention_mask(x, padding_id, block_length, None, extra_tokens=0)
    num_transfer_tokens = get_num_transfer_tokens(block_length, denoising_steps).to(x.device)

    # ===== prefill =====
    with torch.no_grad():
        chunked_prefill(
            teacher_model,
            x,
            chunk_size=512, 
            attention_mask=attn_mask,
            past_key_values=past_key_values
        )

    # ===== 1️⃣ Teacher =====
    # ===== 1️⃣ Teacher =====
    with torch.no_grad():
        teacher_x, attention_mask, replace_mask, replace_indices = prepare_mask_for_generation(
            x.clone(), padding_id, mask_id, block_length)
        teacher_x = teacher_x[replace_mask].reshape(B, -1)
        teacher_logits = None
        for step in range(0, 2):
            mask_index = (teacher_x == mask_id)
            if step == 0:
                sparsity_params = {"step": 0, "dirty_tokens": [[] for _ in range(B)], "sparsity_ratio": 1}
            else:
                sparsity_params = None
            if mask_index.sum() == 0:
                break
            logits = teacher_model(
                teacher_x,
                attention_mask=attention_mask,
                position_ids=replace_indices,
                past_key_values=past_key_values,
                use_cache=True,
                store_kv=False,
                sparsity_params=sparsity_params,
            ).logits
            x0, _, transfer_index = sample_and_select_transfer_tokens(
                logits=logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cur_x=teacher_x,
                mask_index=mask_index,
                num_transfer_tokens=num_transfer_tokens,
                step=step,
                remasking_strategy=remasking_strategy,
                confidence_threshold=confidence_threshold,
                sample_with_temperature_topk_topp=sample_with_temperature_topk_topp,
            )
            teacher_x[transfer_index * mask_index] = x0[transfer_index * mask_index]
            if teacher_logits is None:
                teacher_logits = torch.zeros_like(logits)
            teacher_logits[mask_index] = logits[mask_index]
            if step == 0:
                first_transfer_index = transfer_index * mask_index
            if step == 1:
                second_mask_index = mask_index

    # attn_mask = build_dllm_attention_mask(x, padding_id, block_length, prompt_lengths, extra_tokens=0)
    past_key_values = DynamicCacheWithPadOverwrite(pad_id=padding_id)

    # ===== prefill =====
    with torch.no_grad():
        chunked_prefill(
            model,
            x,
            chunk_size=512, 
            attention_mask=attn_mask,
            past_key_values=past_key_values
        )

    # ===== 2️⃣ Student：sparsity=1 采样 =====
    student_x, attention_mask, replace_mask, replace_indices = prepare_mask_for_generation(
        x.clone(), padding_id, mask_id, block_length)
    student_x = student_x[replace_mask].reshape(B, -1)
    #with torch.no_grad():
    sparsity_params = {"step": 0, "dirty_tokens": [[] for _ in range(B)], "sparsity_ratio": 1}
    logits_sparse1 = model(
        student_x,
        attention_mask=attention_mask,
        position_ids=replace_indices,
        past_key_values=past_key_values,
        use_cache=True,
        store_kv=False,
        sparsity_params=sparsity_params,
    ).logits
    mask_index = (student_x == mask_id)
    student_x[first_transfer_index] = teacher_x[first_transfer_index]

    # ===== 3️⃣ Student：sparsity=100 带梯度 =====
    dirty_tokens = [transfer_index_.nonzero().cpu().numpy()[...,0].tolist() for transfer_index_ in first_transfer_index]
    if sparsity:
        sparsity_params = {"step": 1, "dirty_tokens": dirty_tokens, "sparsity_ratio": train_sparsity_ratio}
    else:
        sparsity_params = None
    sparsity_params = {"step": 1, "dirty_tokens": dirty_tokens, "sparsity_ratio": train_sparsity_ratio}
    
    outputs_student = model(
        student_x,
        attention_mask=attention_mask,
        position_ids=replace_indices,
        past_key_values=past_key_values,
        use_cache=True,
        store_kv=False,
        sparsity_params=sparsity_params,
    )
    student_logits = outputs_student.logits

    # ===== 4️⃣ KL Loss =====
    T = 1.0  # 温度系数，可调，比如 2～5
    teacher_log_probs = F.log_softmax(teacher_logits[second_mask_index].float() / T, dim=-1)
    student_log_probs = F.log_softmax(student_logits[second_mask_index].float() / T, dim=-1)
    kl_loss = F.kl_div(
        student_log_probs, teacher_log_probs, reduction="batchmean", log_target=True
    ) * (T * T)
    # kl_loss = F.cross_entropy(student_logits[second_mask_index], F.softmax(teacher_logits[second_mask_index], dim=-1))
    acc = (teacher_logits[second_mask_index].argmax(-1)==student_logits[second_mask_index].argmax(-1)).float().mean().item()

    sparsity_params = None
    outputs_student = model(
        student_x,
        attention_mask=attention_mask,
        position_ids=replace_indices,
        past_key_values=past_key_values,
        use_cache=True,
        store_kv=False,
        sparsity_params=sparsity_params,
    )
    student_logits = outputs_student.logits

    # ===== 4️⃣ KL Loss =====
    student_log_probs_dense = F.log_softmax(student_logits[second_mask_index].float() / T, dim=-1)
    kl_loss_dense = F.kl_div(
        student_log_probs_dense, teacher_log_probs, reduction="batchmean", log_target=True
    ) * (T * T)
    acc_dense = (teacher_logits[second_mask_index].argmax(-1)==student_logits[second_mask_index].argmax(-1)).float().mean().item()

    kl_loss_agg = (kl_loss * sparse_loss_weight + kl_loss_dense * dense_loss_weight)

    dump_path = os.environ.get("TRADO_TRAIN_REUSE_DUMP")
    if dump_path:
        with torch.no_grad():
            first_counts = first_transfer_index.sum(dim=1).detach().cpu().tolist()
            second_counts = second_mask_index.sum(dim=1).detach().cpu().tolist()
            dirty_lens = [len(x) for x in dirty_tokens]
            dirty_examples = [list(map(int, x[:16])) for x in dirty_tokens[:8]]
            teacher_argmax = teacher_logits[second_mask_index].argmax(-1)
            sparse_argmax = student_logits[second_mask_index].argmax(-1)
            dense_argmax = outputs_student.logits[second_mask_index].argmax(-1)
            _append_jsonl(dump_path, {
                "source": "train",
                "batch_size": int(B),
                "block_length": int(block_length),
                "denoising_steps": int(denoising_steps),
                "confidence_threshold": float(confidence_threshold),
                "train_sparsity_ratio": int(train_sparsity_ratio),
                "first_transfer_counts": [int(x) for x in first_counts],
                "second_mask_counts": [int(x) for x in second_counts],
                "dirty_lens": [int(x) for x in dirty_lens],
                "dirty_examples": dirty_examples,
                "dirty_len_mean": float(sum(dirty_lens) / len(dirty_lens)) if dirty_lens else 0.0,
                "second_mask_total": int(second_mask_index.sum().item()),
                "kl_sparse": float(kl_loss.detach().float().item()),
                "kl_dense": float(kl_loss_dense.detach().float().item()),
                "acc_sparse": float(acc),
                "acc_dense": float(acc_dense),
                "sparse_teacher_match": float((teacher_argmax == sparse_argmax).float().mean().item()) if teacher_argmax.numel() else 0.0,
                "dense_teacher_match": float((teacher_argmax == dense_argmax).float().mean().item()) if teacher_argmax.numel() else 0.0,
            })

    x = torch.cat([x, teacher_x], dim=1)
    return x, teacher_logits, student_logits, kl_loss_agg, acc, acc_dense, kl_loss.item(), kl_loss_dense.item()
