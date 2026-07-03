from collections import deque
import json
import os
import torch
from torch.nn import functional as F
import numpy as np
import time
from jetengine_ext.config import Config
from jetengine_ext.engine.sequence import Sequence, SequenceStatus, RunType
from jetengine_ext.engine.block_manager import BlockManager
from jetengine_ext.layers.sampler import sample_with_temperature_topk_topp
from flashinfer.logits_processor import LogitsPipe, Temperature, Softmax, TopP, TopK, Sample
from tqdm import tqdm

class Scheduler:

    def __init__(self, config: Config,batch_size:int):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mask_token_id = config.mask_token_id
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.cal_blocks = self.block_manager.init_cal_block(batch_size)
        self.running: list[Sequence] = []
        self.sample_pipe = LogitsPipe([
                                Temperature(),      # Scale logits by temperature
                                TopK(),             # Apply top-k filtering
                                Softmax(),          # Convert logits to probabilities
                                TopP(),             # Apply top-p filtering
                            ])
        self.sample_pipe_topk0 = LogitsPipe([
                        Temperature(),      # Scale logits by temperature
                        Softmax(),          # Convert logits to probabilities
                        TopP(),             # Apply top-p filtering
                        ])
        self.token_ids_backend = {}
        self.token_num_backend = {}
        self.block_table_backend = {}
        self.first_unmask_steps_backend = {}
        self.current_block_tensor=None
        self.count=0
        self.last_post_profile = {}
        self.reuse_dump_count = 0

    def _append_reuse_dump(self, payload):
        path = os.environ.get("TRADO_INFER_REUSE_DUMP")
        if not path:
            return
        limit = int(os.environ.get("TRADO_INFER_REUSE_DUMP_LIMIT", "512"))
        if self.reuse_dump_count >= limit:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.reuse_dump_count += 1

    def _append_schedule_reuse_dump(self, path, payload):
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def add(self, seq: Sequence):
        self.running.append(seq)

    def recover_seqs(self,seqs):
        for id in self.token_ids_backend:
            seqs[id].token_ids = self.token_ids_backend[id]
        for id in self.token_num_backend:
            seqs[id].num_tokens = self.token_num_backend[id]
        for id in self.first_unmask_steps_backend:
            seqs[id].first_unmask_steps = self.first_unmask_steps_backend[id]
        for id in self.block_table_backend:
            seqs[id].block_table = self.block_table_backend[id]
        self.token_ids_backend = {}
        self.token_num_backend = {}
        self.first_unmask_steps_backend = {}
        self.block_table_backend = {}

    def is_finished(self):
        return not self.running

    def schedule(self) -> tuple[list[Sequence], RunType] | tuple[None, None]:
        # 1. Schedule new sequences for prefill
        self.count+=1
        prefill_candidates = [s for s in self.running if s.status == SequenceStatus.WAITING]
        if prefill_candidates:
            prefill_batch = []
            # Simple batching: take as many as fit
            for idx,seq in enumerate(prefill_candidates):
                # num_tokens for a waiting seq is its prefill length
                if len(prefill_batch) < self.max_num_seqs and self.block_manager.can_allocate(seq):
                    self.block_manager.allocate(seq)
                    seq.status = SequenceStatus.PREFILLING
                    if seq.sparsity_params is not None:
                        seq.sparsity_params["current_denoising_step"]=seq.current_denoising_step
                    prefill_batch.append(seq)
            if prefill_batch:
                return prefill_batch, RunType.PREFILL
        # 2. If no prefilling, create a DENOISE batch.
        denoise_candidates = [s for s in self.running if s.status == SequenceStatus.DENOISING or s.status == SequenceStatus.SAVING]
        if denoise_candidates:
            denoise_batch = []
            seq_ids=[]
            for seq in denoise_candidates:
                num_new_blocks = seq.num_new_blocks_needed(self.block_manager.block_size)
                # if seq.seq_id == 1 and num_new_blocks>0:
                #     print(1)
                if num_new_blocks>0 and seq.current_denoising_step>0:
                    num_new_blocks = 0
                if seq.current_denoising_step == 0:
                    if seq.sparsity_params is not None:
                        seq.sparsity_params["dirty_tokens"] = []
                        seq.sparsity_params["dirty_token_mask"] = 0
                        seq.sparsity_params["dirty_token_count"] = 0
                if seq.sparsity_params is not None:
                    seq.sparsity_params["current_denoising_step"]=seq.current_denoising_step
                    seq_ids.append(seq.seq_id)
                if len(denoise_batch) < self.max_num_seqs and self.block_manager.can_append_blocks(num_new_blocks):
                    self.block_manager.append_blocks(seq, num_new_blocks)
                    denoise_batch.append(seq)
            if len(denoise_batch)>0 and denoise_batch[0].sparsity_params is not None:
                # no_need_update_kvcache_idx = [idx for idx in range(len(denoise_batch)) if len(denoise_batch[idx].sparsity_params["dirty_tokens"])< \
                #                               denoise_batch[idx].sparsity_params["sparsity_ratio"] and denoise_batch[idx].sparsity_params["current_denoising_step"]>0]
                # need_update_kvcache_idx = [idx for idx in range(len(denoise_batch)) if idx not in no_need_update_kvcache_idx]
                no_need_update_kvcache_idx = []
                need_update_kvcache_idx = []
                need_store_history_idx = []
                schedule_dump = os.environ.get("TRADO_SCHEDULE_REUSE_DUMP")
                
                no_force_update_on_save = os.environ.get("TRADO_NO_FORCE_UPDATE_ON_SAVE", "0") == "1"
                for idx, seq in enumerate(denoise_batch):
                    sp = seq.sparsity_params
                    dirty_count = sp.get("dirty_token_count")
                    if dirty_count is None:
                        dirty_count = len(sp["dirty_tokens"])
                        sp["dirty_token_count"] = dirty_count
                    can_reuse_cache = seq.status == SequenceStatus.DENOISING or (
                        no_force_update_on_save and seq.status == SequenceStatus.SAVING
                    )
                    no_update_cache = (
                        dirty_count < sp["sparsity_ratio"]
                        and sp["current_denoising_step"] > 0
                        and can_reuse_cache
                    )
                    if (
                        os.environ.get("TRADO_FORCE_DENOISE_NO_UPDATE", "0") == "1"
                        and seq.status == SequenceStatus.DENOISING
                        and sp["current_denoising_step"] > 0
                    ):
                        no_update_cache = True
                    if schedule_dump:
                        self._append_schedule_reuse_dump(schedule_dump, {
                            "scheduler_step": int(self.count),
                            "batch_idx": int(idx),
                            "seq_id": int(seq.seq_id),
                            "status": seq.status.name,
                            "current_denoising_step": int(sp["current_denoising_step"]),
                            "dirty_count": int(dirty_count),
                            "sparsity_ratio": int(sp["sparsity_ratio"]),
                            "need_update": not bool(no_update_cache),
                            "no_update": bool(no_update_cache),
                            "no_force_update_on_save": bool(no_force_update_on_save),
                        })
                    if no_update_cache:
                        if seq.status == SequenceStatus.DENOISING:
                            self.token_num_backend[idx]=seq.num_tokens
                            self.token_ids_backend[idx]=seq.token_ids
                            self.block_table_backend[idx]=seq.block_table
                            self.first_unmask_steps_backend[idx]=seq.first_unmask_steps
                            seq.num_tokens_backend = seq.num_tokens
                            seq.block_table = [self.cal_blocks[idx]]
                            seq.num_tokens = 0
                            seq.token_ids = []
                            seq.first_unmask_steps = []
                        no_need_update_kvcache_idx.append(idx)
                    else:
                        need_update_kvcache_idx.append(idx)
                        mask_count = sum(1 for token_id in seq.intermediate_block_tokens if token_id == self.mask_token_id)
                        transfer_budget = 1
                        if seq.current_denoising_step < len(seq.num_transfer_tokens_per_step):
                            transfer_budget = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                        block_finishes_this_step = (
                            seq.status == SequenceStatus.SAVING
                            or mask_count <= max(1, int(transfer_budget))
                            or seq.current_denoising_step + 1 >= seq.denoising_steps
                        )
                        if not block_finishes_this_step:
                            need_store_history_idx.append(idx)

            if len(denoise_batch)>0 and denoise_batch[0].sparsity_params is not None:
                denoise_batch[0].sparsity_params["no_need_update_kvcache_idx"] = no_need_update_kvcache_idx
                denoise_batch[0].sparsity_params["need_update_kvcache_idx"] = need_update_kvcache_idx
                denoise_batch[0].sparsity_params["need_store_history_idx"] = need_store_history_idx
                denoise_batch[0].sparsity_params["seq_ids"] = seq_ids
                denoise_batch[0].sparsity_params["dirty_mask"]=None
            if denoise_batch:
                return denoise_batch, RunType.DENOISE

        return None, None     

    def postprocess(self, seqs: list[Sequence], logits: torch.Tensor, run_type: RunType):
        if run_type == RunType.PREFILL:
            for seq in seqs:
                seq.num_cached_tokens = seq.num_prefill_tokens
                seq.status = SequenceStatus.DENOISING
        elif run_type == RunType.DENOISE:
            post_profile = {
                "post_inner_logits_processor_ms": 0.0,
                "post_inner_sample_gather_ms": 0.0,
                "post_inner_block_tensor_ms": 0.0,
                "post_inner_batched_transfer_ms": 0.0,
                "post_inner_batched_transfer_count": 0.0,
                "post_inner_loop_update_ms": 0.0,
                "post_inner_filter_deallocate_ms": 0.0,
                "post_inner_total_profiled_ms": 0.0,
            }
            start_idx = 0
            cpu_post_sched_ops = os.environ.get("TRADO_GPU_POST_SCHED_OPS", "0") != "1"
            if self.consistent_sampling_params:
                st = time.time()
                if seqs[0].top_k > 0:
                    probs = self.sample_pipe(logits, temperature=seqs[0].temperature, top_k=seqs[0].top_k, top_p=seqs[0].top_p) 
                else:
                    probs = self.sample_pipe_topk0(logits, temperature=seqs[0].temperature, top_p=seqs[0].top_p)
                post_profile["post_inner_logits_processor_ms"] = (time.time() - st) * 1000.0

                st = time.time()
                sampled_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1)
                sampled_x0_2d = sampled_x0.view(len(seqs), seqs[0].block_length)
                sampled_x0_p = torch.gather(probs, -1, sampled_x0.unsqueeze(-1)).squeeze(-1)
                sampled_x0_p_2d = sampled_x0_p.view(len(seqs), seqs[0].block_length)
                if cpu_post_sched_ops:
                    sampled_x0_2d_cpu = sampled_x0_2d.cpu().tolist()
                    sampled_x0_p_2d_cpu = sampled_x0_p_2d.cpu().tolist()
                else:
                    sampled_x0_2d_cpu = None
                    sampled_x0_p_2d_cpu = None
                post_profile["post_inner_sample_gather_ms"] = (time.time() - st) * 1000.0

                st = time.time()
                if cpu_post_sched_ops:
                    current_block_lists = [list(seq.intermediate_block_tokens) for seq in seqs]
                    current_block_tensors = None
                else:
                    current_block_lists = None
                    current_block_tensors = torch.tensor(
                        [seq.intermediate_block_tokens for seq in seqs],
                        device=logits.device,
                    )
                post_profile["post_inner_block_tensor_ms"] = (time.time() - st) * 1000.0
            else:
                current_block_lists = None
                current_block_tensors = None
            if seqs and seqs[0].sparsity_params is not None and "need_update_kvcache_idx" in seqs[0].sparsity_params:
                need_update_idx = seqs[0].sparsity_params["need_update_kvcache_idx"]
                need_update_list = need_update_idx.cpu().tolist() if torch.is_tensor(need_update_idx) else need_update_idx
                need_update_flags = [False] * len(seqs)
                for i in need_update_list:
                    need_update_flags[int(i)] = True
            else:
                need_update_flags = [False] * len(seqs)
            for seq_index,seq in enumerate(seqs):
                # Extract the part of the tensors relevant to this sequence
                if seq.status == SequenceStatus.DENOISING:
                    block_len = seq.block_length
                    if not self.consistent_sampling_params:
                        if seq.top_k > 0:
                            probs = self.sample_pipe(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_k=seq.top_k, top_p=seq.top_p) 
                        else:
                            probs = self.sample_pipe_topk0(logits[start_idx : start_idx + block_len], temperature=seq.temperature, top_p=seq.top_p)
                        st_sample = time.time()
                        seq_x0 = torch.multinomial(probs, num_samples=1).squeeze(-1) 
                        seq_x0_p = torch.gather(probs, -1, seq_x0.unsqueeze(-1)).squeeze(-1)
                        post_profile["post_inner_sample_gather_ms"] += (time.time() - st_sample) * 1000.0
                        current_block_tensor = torch.tensor(seq.intermediate_block_tokens, device=logits.device)
                    else:
                        if cpu_post_sched_ops:
                            seq_x0_list = sampled_x0_2d_cpu[seq_index]
                            seq_x0_p_list = sampled_x0_p_2d_cpu[seq_index]
                            current_block_list = current_block_lists[seq_index]
                            mask_positions = [
                                i for i, token_id in enumerate(current_block_list)
                                if token_id == self.mask_token_id
                            ]
                        else:
                            seq_x0 = sampled_x0_2d[seq_index]
                            seq_x0_p = sampled_x0_p_2d[seq_index]
                            current_block_tensor = current_block_tensors[seq_index]
                            mask_positions = None
                    if self.consistent_sampling_params:
                        if cpu_post_sched_ops:
                            mask_index = None
                        else:
                            mask_index = (current_block_tensor == self.mask_token_id)
                    else:
                        mask_index = (current_block_tensor == self.mask_token_id)
                    num_to_transfer = seq.num_transfer_tokens_per_step[seq.current_denoising_step]
                    
                    if self.consistent_sampling_params and cpu_post_sched_ops:
                        transfer_indices = []
                    else:
                        transfer_index = torch.zeros_like(seq_x0, dtype=torch.bool)
                    
                    if seq.remasking_strategy == 'sequential':
                        if self.consistent_sampling_params:
                            if cpu_post_sched_ops:
                                if mask_positions:
                                    first_mask_pos = min(mask_positions)
                                    end_pos = min(first_mask_pos + num_to_transfer, block_len)
                                    transfer_indices = list(range(first_mask_pos, end_pos))
                            else:
                                if mask_index.any():
                                    first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                                    end_pos = min(first_mask_pos + num_to_transfer, block_len)
                                    transfer_index[first_mask_pos:end_pos] = True
                        else:
                            if mask_index.any():
                                first_mask_pos = mask_index.nonzero(as_tuple=True)[0].min().item()
                                end_pos = min(first_mask_pos + num_to_transfer, block_len)
                                transfer_index[first_mask_pos:end_pos] = True
                    
                    elif 'low_confidence_static' in seq.remasking_strategy:
                        if self.consistent_sampling_params:
                            if cpu_post_sched_ops:
                                transfer_indices = sorted(
                                    mask_positions,
                                    key=lambda i: seq_x0_p_list[i],
                                    reverse=True,
                                )[:num_to_transfer]
                            else:
                                confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                                _, top_indices = torch.topk(confidence, num_to_transfer)
                                transfer_index[top_indices] = True
                        else:
                            confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                            # For dynamic, add threshold logic here if desired
                            _, top_indices = torch.topk(confidence, num_to_transfer)
                            transfer_index[top_indices] = True
                    
                    elif 'low_confidence_dynamic' in seq.remasking_strategy:
                        st=time.time()
                        if self.consistent_sampling_params:
                            if cpu_post_sched_ops:
                                above_threshold = [
                                    i for i in mask_positions
                                    if seq_x0_p_list[i] > seq.dynamic_threshold
                                ]
                                top_indices = None
                                if len(above_threshold) < num_to_transfer:
                                    top_indices = sorted(
                                        mask_positions,
                                        key=lambda i: seq_x0_p_list[i],
                                        reverse=True,
                                    )[:num_to_transfer]
                                    transfer_indices = top_indices
                                else:
                                    transfer_indices = above_threshold
                                num_to_transfer = len(transfer_indices) if len(transfer_indices) > 0 else num_to_transfer
                            else:
                                confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                                transfer_index = torch.where(confidence > seq.dynamic_threshold, True, False)
                                top_indices = None
                                if sum(transfer_index) < num_to_transfer:
                                    _, top_indices = torch.topk(confidence, num_to_transfer)
                                    transfer_index[top_indices] = True
                                num_to_transfer = transfer_index.sum().item() if transfer_index.sum().item() > 0 else num_to_transfer
                        else:
                            confidence = torch.where(mask_index, seq_x0_p, -np.inf)
                            transfer_index = torch.where(confidence > seq.dynamic_threshold, True, False)
                            top_indices=None
                            if sum(transfer_index) < num_to_transfer:
                                _, top_indices = torch.topk(confidence, num_to_transfer)
                                transfer_index[top_indices] = True
                            num_to_transfer = transfer_index.sum().item() if transfer_index.sum().item() > 0 else num_to_transfer
                        if seq.sparsity_params is not None:
                            dump_enabled = bool(os.environ.get("TRADO_INFER_REUSE_DUMP"))
                            if dump_enabled:
                                prev_dirty_len = int(seq.sparsity_params.get("dirty_token_count", len(seq.sparsity_params.get("dirty_tokens", []))))
                            if self.consistent_sampling_params and cpu_post_sched_ops:
                                new_dirty_tokens = transfer_indices
                            elif top_indices is None:
                                new_dirty_tokens = transfer_index.nonzero(as_tuple=True)[0].tolist()
                            else:
                                new_dirty_tokens = top_indices.tolist()
                            dirty_tokens = seq.sparsity_params.setdefault("dirty_tokens", [])
                            if need_update_flags[seq_index]:
                                seq.update_kv_cache=1
                                seq.no_update_kv_cache=0
                                dirty_tokens.clear()
                                dirty_mask_value = 0
                            else:
                                seq.update_kv_cache=0
                                seq.no_update_kv_cache=1
                                dirty_mask_value = int(seq.sparsity_params.get("dirty_token_mask", 0))
                            if new_dirty_tokens:
                                dirty_tokens.extend(new_dirty_tokens)
                                for dirty_idx in new_dirty_tokens:
                                    dirty_mask_value |= 1 << int(dirty_idx)
                            seq.sparsity_params["dirty_token_mask"] = dirty_mask_value
                            seq.sparsity_params["dirty_token_count"] = len(dirty_tokens)
                            if dump_enabled:
                                if self.consistent_sampling_params and cpu_post_sched_ops:
                                    selected_conf = [float(seq_x0_p_list[i]) for i in new_dirty_tokens]
                                    mask_conf = [float(seq_x0_p_list[i]) for i in mask_positions]
                                elif new_dirty_tokens:
                                    selected_conf = seq_x0_p[new_dirty_tokens].detach().float().cpu().tolist()
                                    mask_conf = seq_x0_p[mask_index].detach().float().cpu().tolist()
                                else:
                                    selected_conf = []
                                    mask_conf = seq_x0_p[mask_index].detach().float().cpu().tolist()
                                self._append_reuse_dump({
                                    "source": "infer",
                                    "scheduler_step": int(self.count),
                                    "seq_index": int(seq_index),
                                    "seq_id": int(seq.seq_id),
                                    "status": str(seq.status),
                                    "block_length": int(seq.block_length),
                                    "current_denoising_step_before_post": int(seq.current_denoising_step),
                                    "global_denoising_step_before_post": int(seq.global_denoising_step),
                                    "dynamic_threshold": float(seq.dynamic_threshold),
                                    "sparsity_ratio": int(seq.sparsity_params.get("sparsity_ratio", 0)),
                                    "need_update": bool(seq.update_kv_cache),
                                    "no_need_update": bool(seq.no_update_kv_cache),
                                    "need_update_batch_size": int(sum(need_update_flags)),
                                    "prev_dirty_len": prev_dirty_len,
                                    "new_dirty_len": int(len(new_dirty_tokens)),
                                    "new_dirty_tokens": [int(x) for x in new_dirty_tokens[:16]],
                                    "dirty_len_after": int(seq.sparsity_params.get("dirty_token_count", len(seq.sparsity_params["dirty_tokens"]))),
                                    "mask_count": int(len(mask_positions) if self.consistent_sampling_params else mask_index.sum().item()),
                                    "num_to_transfer": int(num_to_transfer),
                                    "selected_conf": [float(x) for x in selected_conf[:16]],
                                    "selected_conf_mean": float(sum(selected_conf) / len(selected_conf)) if selected_conf else 0.0,
                                    "mask_conf_mean": float(sum(mask_conf) / len(mask_conf)) if mask_conf else 0.0,
                                    "mask_conf_max": float(max(mask_conf)) if mask_conf else 0.0,
                                    "mask_conf_min": float(min(mask_conf)) if mask_conf else 0.0,
                                })
                        else:
                            seq.update_kv_cache=1
                            seq.no_update_kv_cache=0
                        post_profile["post_inner_batched_transfer_ms"] += (time.time()-st) * 1000.0
                        post_profile["post_inner_batched_transfer_count"] += float(num_to_transfer)
                    elif 'entropy_bounded' in seq.remasking_strategy:
                        block_probs = probs[start_idx : start_idx + block_len]
                        P = block_probs[mask_index]
                        eps = 1e-12
                        entropies = -(P.clamp_min(eps) * (P.clamp_min(eps)).log()).sum(dim=-1)
                        ent_sorted, order = torch.sort(entropies, dim=0, descending=False)
                        cumsum = torch.cumsum(ent_sorted, dim=0)
                        k = torch.searchsorted(cumsum, torch.tensor(seq.eb_threshold, device=P.device), right=False).item()
                        if k == 0:
                            k = 1
                        # print(k)
                        selected_token_indices = mask_index.nonzero(as_tuple=True)[0][order[:k]]
                        # print(selected_token_indices)
                        transfer_index[selected_token_indices] = True
                        num_to_transfer = k

                    # update
                    st=time.time()
                    if self.consistent_sampling_params and cpu_post_sched_ops:
                        new_block_list = current_block_list
                        original_indices = list(transfer_indices)
                        accepted_tokens = [seq_x0_list[idx] for idx in original_indices]
                    else:
                        new_block_list = current_block_tensor.tolist()
                        accepted_tokens = seq_x0[transfer_index].tolist()
                        original_indices = transfer_index.nonzero(as_tuple=True)[0].tolist()





                    # newly added
                    if seq.block_first_unmask_steps is None or len(seq.block_first_unmask_steps) != block_len:
                        seq.block_first_unmask_steps = [0] * block_len
                    first_time_global = seq.global_denoising_step + 1
                    for idx in original_indices:
                        if seq.block_first_unmask_steps[idx] == 0:
                            seq.block_first_unmask_steps[idx] = first_time_global
                    

                    for idx, token in zip(original_indices, accepted_tokens):
                        new_block_list[idx] = token
                    seq.intermediate_block_tokens = new_block_list
                    
                    seq.current_denoising_step += 1
                    seq.global_denoising_step += 1
                    
                    # Check if block is fully denoised
                    is_fully_denoised = (self.mask_token_id not in seq.intermediate_block_tokens) or \
                                        (seq.current_denoising_step >= seq.denoising_steps)

                    if is_fully_denoised:
                        # Block is done, commit it and check if generation is finished
                        seq.status = SequenceStatus.FINISHED if seq.is_finished else SequenceStatus.SAVING
                    seq.num_to_transfer = num_to_transfer
                    post_profile["post_inner_loop_update_ms"] += (time.time()-st) * 1000.0
                    
                elif seq.status == SequenceStatus.SAVING:
                    # If saving, commit the block and start a new one
                    seq.commit_block(seq.intermediate_block_tokens)
                    seq.num_to_transfer = 0
                    if not seq.is_finished:
                        seq.start_new_block()

                start_idx += seq.block_length
        # Filter out finished sequences from the running list
        st_filter = time.time()
        finished_seqs = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished_seqs:
            self.block_manager.deallocate(seq)
        if run_type == RunType.DENOISE:
            post_profile["post_inner_filter_deallocate_ms"] = (time.time() - st_filter) * 1000.0
            post_profile["post_inner_total_profiled_ms"] = (
                post_profile["post_inner_logits_processor_ms"]
                + post_profile["post_inner_sample_gather_ms"]
                + post_profile["post_inner_block_tensor_ms"]
                + post_profile["post_inner_batched_transfer_ms"]
                + post_profile["post_inner_loop_update_ms"]
                + post_profile["post_inner_filter_deallocate_ms"]
            )
            self.last_post_profile = post_profile
        else:
            self.last_post_profile = {}
