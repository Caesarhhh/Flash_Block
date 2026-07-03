import pickle
import os
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import time
from jetengine_ext.config import Config
from jetengine_ext.engine.sequence import Sequence, RunType, SequenceStatus
from jetengine_ext.models.sdar import SDARForCausalLM
from jetengine_ext.models.sdar_moe import SDARMoeForCausalLM
from jetengine_ext.utils.context import set_context, get_context, reset_context
from jetengine_ext.utils.loader import load_model
import struct


class DenoiseSeqView:
    def __init__(self, state):
        self.num_tokens = state["num_tokens"]
        self.num_tokens_backend = state["num_tokens_backend"]
        self.block_table = state["block_table"]
        self.intermediate_block_tokens = state["intermediate_block_tokens"]
        self.first_unmask_steps = state.get("first_unmask_steps", [])
        self.block_first_unmask_steps = state.get("block_first_unmask_steps")
        self.block_length = state.get("block_length", len(self.intermediate_block_tokens))
        self.sparsity_params = state["sparsity_params"]

    def __len__(self):
        return self.num_tokens


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event],batch_size: int = 128):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.batch_size = batch_size
        self.read_time=0

        port = os.environ.get("MASTER_PORT",2333)

        dist.init_process_group("nccl", f"tcp://localhost:{port}", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        if "sdar" in hf_config.model_type and "moe" in hf_config.model_type:
            self.model = SDARMoeForCausalLM(hf_config)
        elif "sdar" in hf_config.model_type:
            self.model = SDARForCausalLM(hf_config)
        else:
            raise ValueError(f"Unsupported model type: {hf_config.model_type}")
        load_model(self.model, config.model)
        # Sampler is removed from here
        self.warmup_model()
        self.allocate_kv_cache()
        # CUDA graph capture for block diffusion is complex and omitted for this example
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        self.n=0

        if self.world_size > 1:
            shm_name = "jetengineshm"
            if rank == 0:
                # Create (and clean up stale) shared memory
                try:
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**26)
                except FileExistsError:
                    try:
                        stale = SharedMemory(name=shm_name)
                        stale.close()
                        stale.unlink()
                    except FileNotFoundError:
                        pass
                    self.shm = SharedMemory(name=shm_name, create=True, size=2**26)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=shm_name)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args, read_time = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        st=time.time()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, args = self._unpack_call_from_shm(pickle.loads(self.shm.buf[4:n+4]))
        self.event.clear()
        read_time=time.time()-st
        self.shm.buf[n+4:n+12] = struct.pack("d", read_time)
        # print(f"[read_shm] payload size = {n} bytes {read_time}")
        self.read_time=read_time
        self.n=n
        return method_name, args,read_time
    
    def get_read_time(self):
        try:
            read_time = struct.unpack("d", self.shm.buf[self.n+4:self.n+12])[0]
        except:
            return 0
        return read_time

    def _pack_sparsity_params_for_shm(self, sp):
        if sp is None:
            return None
        packed = {}
        for key, value in sp.items():
            if key.startswith("_"):
                continue
            if key in ("need_update_kvcache_idx", "no_need_update_kvcache_idx", "seq_ids"):
                if torch.is_tensor(value):
                    packed[key] = value.detach().cpu().tolist()
                else:
                    packed[key] = list(value)
            elif key == "dirty_tokens":
                packed[key] = list(value)
            elif key == "is_saving":
                packed[key] = bool(value)
            else:
                packed[key] = value
        return packed

    def _pack_sparsity_params_for_denoise_shm(self, sp, include_batch_indices=False):
        if sp is None:
            return None
        packed = {
            "dirty_token_mask": int(sp.get("dirty_token_mask", 0)),
            "dirty_token_count": int(sp.get("dirty_token_count", len(sp.get("dirty_tokens", [])))),
        }
        if packed["dirty_token_mask"] == 0 and sp.get("dirty_tokens"):
            dirty_mask_value = 0
            for dirty_idx in sp.get("dirty_tokens", []):
                dirty_mask_value |= 1 << int(dirty_idx)
            packed["dirty_token_mask"] = dirty_mask_value
        if include_batch_indices:
            for key in ("need_update_kvcache_idx", "no_need_update_kvcache_idx", "need_store_history_idx"):
                value = sp.get(key, [])
                if torch.is_tensor(value):
                    packed[key] = value.detach().cpu().tolist()
                else:
                    packed[key] = list(value)
        return packed

    def _unpack_sparsity_params_from_shm(self, sp):
        if sp is None:
            return None
        return dict(sp)

    def _pack_denoise_seqs_for_shm(self, seqs):
        packed_seqs = []
        for idx, seq in enumerate(seqs):
            state = {
                "num_tokens": seq.num_tokens,
                "num_tokens_backend": seq.num_tokens_backend,
                "block_table": list(seq.block_table),
                "intermediate_block_tokens": list(seq.intermediate_block_tokens),
            }
            state["sparsity_params"] = self._pack_sparsity_params_for_denoise_shm(
                seq.sparsity_params,
                include_batch_indices=(idx == 0),
            )
            packed_seqs.append(state)
        return packed_seqs

    def _unpack_denoise_seqs_from_shm(self, packed_seqs):
        seqs = []
        for state in packed_seqs:
            state = dict(state)
            state["sparsity_params"] = self._unpack_sparsity_params_from_shm(state["sparsity_params"])
            seqs.append(DenoiseSeqView(state))
        return seqs

    def _pack_call_for_shm(self, method_name, args):
        if method_name == "run" and len(args) >= 2 and args[1] == RunType.DENOISE:
            seqs, run_type, *rest = args
            return [method_name, "__denoise_slim_v1__", self._pack_denoise_seqs_for_shm(seqs), run_type, *rest]
        return [method_name, *args]

    def _unpack_call_from_shm(self, payload):
        method_name, *args = payload
        if method_name == "run" and args and args[0] == "__denoise_slim_v1__":
            _, packed_seqs, run_type, *rest = args
            args = [self._unpack_denoise_seqs_from_shm(packed_seqs), run_type, *rest]
        return method_name, args

    def write_shm(self, method_name, *args):
        st=time.time()
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps(self._pack_call_for_shm(method_name, args))
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # print(f"[write_shm] payload size = {n} bytes {write_time}")
        self.n=n
        for event in self.event:
            event.set()
        write_time=time.time()-st
        return write_time,n

    def call(self, method_name, *args):
        write_time=0
        n=0
        if self.world_size > 1 and self.rank == 0:
            write_time,n=self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args),write_time,n

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        warmup_len = int(os.environ.get("TRADO_WARMUP_MODEL_LEN", max_model_len))
        max_model_len = max(1, min(max_model_len, warmup_len))
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len, self.config.mask_token_id) for _ in range(num_seqs)]
        self.run(seqs, RunType.PREFILL)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        attn_output_past_kwargs = {"dtype": torch.float32}
        self.attn_output_past = torch.zeros(
            hf_config.num_hidden_layers,
            self.batch_size,
            config.block_length,
            hf_config.num_attention_heads // self.world_size,
            hf_config.head_dim,
            **attn_output_past_kwargs,
        )
        self.logsumexp = torch.zeros(hf_config.num_hidden_layers, self.batch_size, config.block_length, hf_config.num_attention_heads // self.world_size, 1, dtype=torch.float32)
        self.logsumexp_all = torch.zeros(hf_config.num_hidden_layers, self.batch_size, config.block_length, hf_config.num_attention_heads // self.world_size, 1, dtype=torch.float32)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
            if hasattr(module, "attn_output_past") and hasattr(module, "logsumexp"):
                module.attn_output_past = self.attn_output_past[layer_id]
                module.logsumexp = self.logsumexp[layer_id]
                module.logsumexp_all = self.logsumexp_all[layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if max_len == 0: return None
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(block_tables, dtype=torch.int32).cuda()

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids, positions, cu_seqlens_q, slot_mapping, is_last_step = [], [], [0], [], []
        max_seqlen_q = 0
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq.token_ids)
            positions.extend(range(seqlen))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen)
            max_seqlen_q = max(max_seqlen_q, seqlen)
            is_last_step.append(False)
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            # Slot mapping for prefill
            if not seq.block_table:
                continue
            for i in range(seqlen):
                block_idx = i // self.block_size 
                block_offset = i % self.block_size 
                physical_block_id = seq.block_table[block_idx]
                slot = physical_block_id * self.block_size + block_offset
                slot_mapping.append(slot)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(positions, dtype=torch.int64).cuda()
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).cuda()
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32).cuda()
        set_context(
            run_type=RunType.PREFILL,
            cu_seqlens_q=cu_seqlens_q, 
            cu_seqlens_k=cu_seqlens_q, 
            max_seqlen_q=max_seqlen_q, 
            max_seqlen_k=max_seqlen_q, 
            slot_mapping=slot_mapping, 
            is_last_denoise_step=is_last_step, # <-- Pass the new flag
            block_length=self.config.block_length
        )
        return input_ids, positions

    def prepare_denoise(self, seqs: list[Sequence]):
        input_ids, positions = [], []
        cached_lens = []
        delta_cached_lens = []
        
        for seq in seqs:
            # The query is the current intermediate block
            q_tokens = seq.intermediate_block_tokens
            q_len = len(q_tokens)
            
            # The context (key/value) is the confirmed part of the sequence
            k_len = len(seq)
            # if k_len == 0:
            #     print(0)
            
            input_ids.extend(q_tokens)
            # Positions are global
            if k_len == 0:
                positions.extend(range(seq.num_tokens_backend, seq.num_tokens_backend + q_len))
                delta_cached_lens.append(seq.num_tokens_backend)
                # k_len=seq.num_tokens_backend
            else:
                positions.extend(range(k_len, k_len + q_len))
                delta_cached_lens.append(0)
            cached_lens.append(k_len)

        input_ids = torch.tensor(input_ids, dtype=torch.int64).cuda()
        positions = torch.tensor(positions, dtype=torch.int64).cuda()
        cached_lens = torch.tensor(cached_lens, dtype=torch.int32).cuda()
        delta_cached_lens = torch.tensor(delta_cached_lens, dtype=torch.int32).cuda()
        block_tables = self.prepare_block_tables(seqs)
        need_update_kvcache_mask = None
        need_store_history_mask = None
        dirty_mask=None
        has_sparsity = seqs[0].sparsity_params is not None
        no_need_update_count = 0
        need_store_history_count = 0
        if seqs[0].sparsity_params is not None:
            sp0 = seqs[0].sparsity_params
            need_update_idx = sp0.get("need_update_kvcache_idx", [])
            no_need_update_idx = sp0.get("no_need_update_kvcache_idx", [])
            need_store_history_idx = sp0.get("need_store_history_idx", need_update_idx)
            if not torch.is_tensor(need_update_idx):
                need_update_idx = torch.as_tensor(need_update_idx, device="cuda", dtype=torch.long)
            else:
                need_update_idx = need_update_idx.to(device="cuda", dtype=torch.long)
            if not torch.is_tensor(no_need_update_idx):
                no_need_update_idx = torch.as_tensor(no_need_update_idx, device="cuda", dtype=torch.long)
            else:
                no_need_update_idx = no_need_update_idx.to(device="cuda", dtype=torch.long)
            if not torch.is_tensor(need_store_history_idx):
                need_store_history_idx = torch.as_tensor(need_store_history_idx, device="cuda", dtype=torch.long)
            else:
                need_store_history_idx = need_store_history_idx.to(device="cuda", dtype=torch.long)
            sp0["_need_update_kvcache_idx_cuda"] = need_update_idx
            sp0["_no_need_update_kvcache_idx_cuda"] = no_need_update_idx
            sp0["_need_store_history_idx_cuda"] = need_store_history_idx
            no_need_update_count = int(no_need_update_idx.numel())
            need_store_history_count = int(need_store_history_idx.numel())
            num_attn_heads = self.attn_output_past.shape[-2]
            if no_need_update_count == 0:
                if need_store_history_count > 0:
                    need_store_history_mask = torch.zeros((len(seqs), 1, 1, 1), dtype=torch.bool, device="cuda")
                    need_store_history_mask[need_store_history_idx] = True
                    need_store_history_mask = need_store_history_mask.repeat(
                        1, num_attn_heads, seqs[0].block_length, 1
                    ).contiguous()
            else:
                need_update_kvcache_mask = torch.zeros((len(seqs), 1, 1, 1), dtype=torch.bool, device="cuda")
                need_update_kvcache_mask[need_update_idx]=True
                need_store_history_mask = torch.zeros_like(need_update_kvcache_mask)
                need_store_history_mask[need_store_history_idx]=True

                dirty_mask = torch.zeros((len(seqs), seqs[0].block_length,1,1), dtype=torch.bool, device="cuda")

                all_b = []
                all_h = []
                for b, seq in enumerate(seqs):
                    sp = seq.sparsity_params
                    dirty_mask_value = int(sp.get("dirty_token_mask", 0))
                    if dirty_mask_value:
                        for h in range(seq.block_length):
                            if dirty_mask_value & (1 << h):
                                all_b.append(b)
                                all_h.append(h)
                        continue
                    dirty = sp.get("dirty_tokens", [])
                    if dirty:
                        all_b.extend([b] * len(dirty))
                        all_h.extend(dirty)
                if all_b:
                    dirty_mask[torch.tensor(all_b, device="cuda"),
                         torch.tensor(all_h, device="cuda")] = True
                dirty_mask=dirty_mask.repeat(1,1,num_attn_heads,1).permute(0,2,1,3).contiguous()
                need_update_kvcache_mask=need_update_kvcache_mask.repeat(1,num_attn_heads,seqs[0].block_length,1).contiguous()
                need_store_history_mask=need_store_history_mask.repeat(1,num_attn_heads,seqs[0].block_length,1).contiguous()
            
        set_context(
            run_type=RunType.DENOISE,
            context_lens=cached_lens,
            delta_cached_lens=delta_cached_lens,
            block_tables=block_tables,
            block_length=self.config.block_length,
            need_update_kvcache_mask=need_update_kvcache_mask,
            need_store_history_mask=need_store_history_mask,
            dirty_mask=dirty_mask,
            has_sparsity=has_sparsity,
            no_need_update_kvcache_count=no_need_update_count,
            need_store_history_count=need_store_history_count,
            all_update_no_store_fastpath=has_sparsity and no_need_update_count == 0 and need_store_history_count == 0,
            all_update_store_only=has_sparsity and no_need_update_count == 0,
        )
        
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, sparsity_params=None):
        return self.model.compute_logits(self.model(input_ids, positions, sparsity_params=sparsity_params))

    def run(self, seqs: list[Sequence], run_type: RunType, sparsity_params=None) -> torch.Tensor:
        profile_sync = os.environ.get("TRADO_SYNC_PROFILE", "0") == "1"
        prepare_ms = forward_sync_ms = attention_ms = 0.0
        if profile_sync:
            torch.cuda.synchronize()
            prepare_start = time.perf_counter()
        if run_type == RunType.PREFILL:
            input_ids, positions = self.prepare_prefill(seqs)
        elif run_type == RunType.DENOISE:
            input_ids, positions = self.prepare_denoise(seqs)
        else:
            return None
        if profile_sync:
            torch.cuda.synchronize()
            prepare_ms = (time.perf_counter() - prepare_start) * 1000.0
            forward_start = time.perf_counter()
        logits = self.run_model(input_ids, positions, sparsity_params=[seq.sparsity_params for seq in seqs])
        if profile_sync:
            torch.cuda.synchronize()
            forward_sync_ms = (time.perf_counter() - forward_start) * 1000.0
            context = get_context()
            if context.profile_attention_events:
                attention_ms = sum(
                    start.elapsed_time(end)
                    for start, end in context.profile_attention_events
                )
            else:
                attention_ms = context.profile_attention_s * 1000.0
            local_non_attention_ms = max(0.0, forward_sync_ms - attention_ms)
            profile_tensor = torch.tensor(
                [prepare_ms, forward_sync_ms, attention_ms, local_non_attention_ms],
                dtype=torch.float64,
                device="cuda",
            )
            if self.world_size > 1:
                dist.all_reduce(profile_tensor, op=dist.ReduceOp.MAX)
            prepare_ms, forward_sync_ms, attention_ms, non_attention_local_ms = profile_tensor.cpu().tolist()
            profile_info = {
                "prepare_ms": prepare_ms,
                "forward_sync_ms": forward_sync_ms,
                "attention_ms": attention_ms,
                "non_attention_forward_ms": max(0.0, forward_sync_ms - attention_ms),
                "non_attention_local_ms": non_attention_local_ms,
            }
            if os.environ.get("TRADO_DETAIL_PROFILE", "0") == "1" and context.profile_detail_s:
                detail_names = sorted(context.profile_detail_s)
                detail_tensor = torch.tensor(
                    [context.profile_detail_s[name] * 1000.0 for name in detail_names],
                    dtype=torch.float64,
                    device="cuda",
                )
                if self.world_size > 1:
                    dist.all_reduce(detail_tensor, op=dist.ReduceOp.MAX)
                for name, value in zip(detail_names, detail_tensor.cpu().tolist()):
                    profile_info[name.replace("_s", "_ms")] = value
        reset_context()
        if profile_sync:
            return (logits, profile_info) if self.rank == 0 else None
        return logits if self.rank == 0 else None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 256)
        max_global_bs = max_bs * self.config.block_length
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_global_bs, dtype=torch.int64)
        positions = torch.zeros(max_global_bs, dtype=torch.int64)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_global_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(run_type=RunType.DENOISE, context_lens=context_lens[:bs], block_tables=block_tables[:bs], block_length=self.config.block_length)
            global_bs = bs * self.config.block_length
            outputs[:global_bs] = self.model(input_ids[:global_bs], positions[:global_bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:global_bs] = self.model(input_ids[:global_bs], positions[:global_bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
