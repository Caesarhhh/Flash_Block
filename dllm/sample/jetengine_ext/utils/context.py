from dataclasses import dataclass, field
from typing import List
import torch

from jetengine_ext.engine.sequence import RunType

@dataclass
class Context:
    run_type: RunType | None = None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_last_denoise_step: List[bool] = field(default_factory=lambda: [False])
    block_length: int = 4
    dirty_mask: torch.Tensor | None = None
    need_update_kvcache_mask: torch.Tensor | None = None
    need_store_history_mask: torch.Tensor | None = None
    delta_cached_lens: torch.Tensor | None = None
    has_sparsity: bool = False
    no_need_update_kvcache_count: int = 0
    need_store_history_count: int = 0
    all_update_no_store_fastpath: bool = False
    all_update_store_only: bool = False
    profile_attention_s: float = 0.0
    profile_attention_events: list = field(default_factory=list)
    profile_detail_s: dict = field(default_factory=dict)

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(run_type, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, is_last_denoise_step=[False], block_length=4, dirty_mask=None,need_update_kvcache_mask=None,need_store_history_mask=None,delta_cached_lens=None, has_sparsity=False, no_need_update_kvcache_count=0, need_store_history_count=0, all_update_no_store_fastpath=False, all_update_store_only=False):
    global _CONTEXT
    _CONTEXT = Context(
        run_type=run_type,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        is_last_denoise_step=is_last_denoise_step,
        block_length=block_length,
        dirty_mask=dirty_mask,
        need_update_kvcache_mask=need_update_kvcache_mask,
        need_store_history_mask=need_store_history_mask,
        delta_cached_lens=delta_cached_lens,
        has_sparsity=has_sparsity,
        no_need_update_kvcache_count=no_need_update_kvcache_count,
        need_store_history_count=need_store_history_count,
        all_update_no_store_fastpath=all_update_no_store_fastpath,
        all_update_store_only=all_update_store_only,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
