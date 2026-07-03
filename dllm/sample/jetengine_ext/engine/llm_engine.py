import atexit
import os
from dataclasses import fields
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
# Added imports for profiling
import torch
import time
from torch import nn
from contextlib import nullcontext
import torch.profiler as torch_profiler

from jetengine_ext.config import Config
from jetengine_ext.sampling_params import SamplingParams
from jetengine_ext.engine.sequence import Sequence, RunType
from jetengine_ext.engine.scheduler import Scheduler
from jetengine_ext.engine.model_runner import ModelRunner
from jetengine_ext.utils.loader import load_from_hf_model

import torch

def move_to_end_torch(max_value, move_indices):
    move = torch.tensor(move_indices, dtype=torch.long)
    
    # 生成 0..max_value-1
    all_idx = torch.arange(max_value, dtype=torch.long, device=move.device)

    # 计算 mask：哪些 index 需要移动
    mask = torch.zeros(max_value, dtype=torch.bool, device=move.device)
    mask[move] = True

    # 前半部分：不在 move 内的
    front = all_idx[~mask]

    # 后半部分：move_indices 的顺序保持不变
    back = move

    return torch.cat([front, back])

def extract_last_boxed(text: str):
    """
    提取最后一个完整闭合的 \\boxed{...} 中的内容。
    如果末尾有未闭合的 \\boxed{...}，跳过它并继续向前找。
    若不存在完整 boxed 则返回 None。
    """
    target = r"\boxed{"
    search_end = len(text)

    while True:
        pos = text.rfind(target, 0, search_end)
        if pos == -1:
            return None

        i = pos + len(target)
        depth = 1
        start = i

        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i]
            i += 1

        search_end = pos

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        batch_size=kwargs.get("batch_size",128)
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events,batch_size)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        config.eos = self.tokenizer.eos_token_id
        config.mask_token_id = self.tokenizer.mask_token_id if self.tokenizer.mask_token_id is not None else self.tokenizer.pad_token_id
        assert config.mask_token_id is not None, "Model tokenizer must have a mask_token_id or pad_token_id"

        self.config = config
        self.scheduler = Scheduler(config,batch_size)
        self.scheduler.consistent_sampling_params = False
        self.flag=time.time()
        atexit.register(self.exit)

    def offload_parameters(self, include_buffers: bool = False):
        """
        Replace all parameter (and buffer) storages with meta tensors.
        Keeps shapes/dtypes, frees GPU/CPU memory.
        """

        def offload_parameters_keep_buffers(model: torch.nn.Module):
            """
            Move *parameters* to meta to free memory while keeping buffers unchanged.
            Works for any module tree.
            """
            # 1) Snapshot real buffers (module reference + buffer name + tensor)
            saved_buffers = []
            for mod in model.modules():
                for bname, buf in list(mod._buffers.items()):
                    if buf is not None:
                        saved_buffers.append((mod, bname, buf))

            # 2) Move everything to meta
            model.to_empty(device=torch.device("meta"))

            # 3) Restore the saved, real buffers
            for mod, bname, buf in saved_buffers:
                # Reattach the original tensor (device/dtype preserved)
                mod._buffers[bname] = buf

            torch.cuda.empty_cache()
        if include_buffers:
            self.model_runner.model.to_empty(device=torch.device("meta"))
        else:
            offload_parameters_keep_buffers(self.model_runner.model)

        print("Successfully cleaned old parameters (buffers kept)." if not include_buffers
              else "Successfully cleaned old parameters and buffers.")

    def reload_parameters(self, hf_model: nn.Module):
        load_from_hf_model(self.model_runner.model, hf_model=hf_model)

    def exit(self):
        if hasattr(self, "model_runner"):
            self.model_runner.call("exit")
            del self.model_runner
        for p in getattr(self, "ps", []):
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams, sparsity_ratio: float = 0):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        if isinstance(prompt, list):
            if self.tokenizer.pad_token_id in prompt:
                start = prompt.index(self.tokenizer.pad_token_id) + 1
                prompt = prompt[start:]
        seq = Sequence(prompt, self.config.mask_token_id, sampling_params, sparsity_ratio)
        seq.eos_token_id = self.tokenizer.eos_token_id
        self.scheduler.add(seq)

    def step(self):
        st=time.time()
        scheduled_seqs, run_type = self.scheduler.schedule()
        schedule_time = time.time()-st
        if scheduled_seqs is None:
            return [], 0, 0, 0, 0, 0, 0, 0, schedule_time, 0 # Nothing to run

        # print(f"per step{time.time()-self.flag}")
        self.flag=time.time()
        run_result,write_time,n = self.model_runner.call("run", scheduled_seqs, run_type)
        profile_info = {}
        if isinstance(run_result, tuple) and len(run_result) == 2 and isinstance(run_result[1], dict):
            logits, profile_info = run_result
        else:
            logits = run_result
        run_time=time.time()-self.flag-write_time
        # print(f"run {run_time}")
        self.scheduler.recover_seqs(scheduled_seqs)
        st=time.time()
        self.scheduler.postprocess(scheduled_seqs, logits, run_type)
        post_schedule_time = time.time()-st
        if profile_info:
            profile_info.update(getattr(self.scheduler, "last_post_profile", {}) or {})
        
        #finished_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_seqs if seq.is_finished]
        
        finished_outputs = [
            (seq.seq_id, seq.completion_token_ids, seq.first_unmask_steps,idx)
            for idx,seq in enumerate(scheduled_seqs)
            if seq.is_finished
        ]

        # [extract_last_boxed(self.tokenizer.decode(scheduled_seqs[i].token_ids)) for i in range(len(scheduled_seqs))]
        num_tokens = [self.scheduler.running[i].num_to_transfer if hasattr(self.scheduler.running[i], 'num_to_transfer') else 0 for i in range(len(self.scheduler.running))]
        if scheduled_seqs[0].sparsity_params is not None and "need_update_kvcache_idx" in scheduled_seqs[0].sparsity_params:
            try:
                no_update_kv_cache=sum([len(scheduled_seqs[i].token_ids) for i in scheduled_seqs[0].sparsity_params["no_need_update_kvcache_idx"]])
                update_kv_cache=sum([len(scheduled_seqs[i].token_ids) for i in scheduled_seqs[0].sparsity_params["need_update_kvcache_idx"]])
            except:
                no_update_kv_cache=0
                update_kv_cache=0
        else:
            no_update_kv_cache=0
            update_kv_cache=0
        result = (finished_outputs, sum(num_tokens), update_kv_cache, no_update_kv_cache, sum([len(scheduled_seq.token_ids) for scheduled_seq in scheduled_seqs]),run_time,write_time,n,schedule_time,post_schedule_time)
        if profile_info:
            return result + (profile_info,)
        return result

    def is_finished(self):
        return self.scheduler.is_finished()


    def _clean_token_ids(self, token_ids):
        # Accept tensors, numpy ints, etc.
        try:
            token_ids = list(token_ids)
        except Exception:
            token_ids = [token_ids]
        
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        mask_id = getattr(self.config, "mask_token_id", None)

        cleaned = []
        for t in token_ids:
            if t is None or t < 0 or t == mask_id or t >= vocab_size:
                if t not in special_ids:
                    cleaned.append(0)
                    continue
            cleaned.append(t)
        return cleaned

    def _safe_decode(self, token_ids):
        ids = self._clean_token_ids(token_ids)
        # skip_special_tokens can be True or False; doesn't affect the None issue
        return self.tokenizer.decode(ids, skip_special_tokens=False)
    


    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
    ) -> list[str]:
        # ... (This method remains largely the same, but the progress bar will update differently) ...
        # The logic inside the `while not self.is_finished()` loop correctly calls `self.step()`
        # and collects outputs.
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
            self.scheduler.consistent_sampling_params = True
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        
        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            while not self.is_finished():
                step_result = self.step()
                output, num_processed = step_result[0], step_result[1]
                if profile:
                    prof.step()
                
                #for seq_id, token_ids in output:
                #    outputs[seq_id] = token_ids
                for seq_id, token_ids, unmask_times in output:
                    outputs[seq_id] = {"token_ids": token_ids, "unmask_times": unmask_times}
                    if use_tqdm:
                        pbar.update(1)

        #outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        #outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        outputs = [
            {
                "text": self._safe_decode(item["token_ids"]),
                "token_ids": self._clean_token_ids(item["token_ids"]),
                "first_unmask_times": item["unmask_times"],   # 与 token_ids 等长
            }
            for item in outputs
        ]

        if use_tqdm:
            pbar.close()
        return outputs

    def generate_streaming(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        max_active: int | None = None,
        use_tqdm: bool = True,
        # New optional profiling controls
        profile: bool = False,
        profile_dir: str | None = None,
        sparsity_ratio: float | None = None
    ) -> list[str]:
        """
        Stream prompts through the engine while keeping up to `max_active` sequences running.
        As sequences finish, new prompts are added from the pending list to maximize GPU utilization.
        """
        if sparsity_ratio is None:
            sparsity_ratio = 0
        total = len(prompts)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * total
            self.scheduler.consistent_sampling_params = True

        if max_active is None:
            max_active = getattr(self.scheduler, "max_num_seqs", 32)

        if use_tqdm:
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)

        outputs: dict[int, list[int]] = {}
        pending_idx = 0

        # Prime initial requests up to capacity
        initial = min(max_active, total)
        for i in range(initial):
            self.add_request(prompts[i], sampling_params[i], sparsity_ratio)
        pending_idx = initial

        # Setup profiler context
        activities = [torch_profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch_profiler.ProfilerActivity.CUDA)
        trace_dir = profile_dir or "profiler_traces"
        prof_ctx = (
            torch_profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch_profiler.tensorboard_trace_handler(trace_dir),
            )
            if profile else nullcontext()
        )

        with prof_ctx as prof:
            while not self.is_finished() or pending_idx < total:
                # Top up to capacity before each step
                running = getattr(self.scheduler, "running", [])
                deficit = max_active - len(running)
                while deficit > 0 and pending_idx < total:
                    self.add_request(prompts[pending_idx], sampling_params[pending_idx], sparsity_ratio)
                    pending_idx += 1
                    deficit -= 1
                # [extract_last_boxed(self.tokenizer.decode(outputs[seq_id]["token_ids"])) for seq_id in sorted(outputs)]
                step_result = self.step()
                output, num_processed = step_result[0], step_result[1]
                if sparsity_ratio>0:
                    if len(output)>0:
                        refresh_cache_idxs = move_to_end_torch(max_active,[output_[3] for output_ in output])
                        refresh_cache_idxs = refresh_cache_idxs.to(self.model_runner.attn_output_past.device)
                        self.model_runner.attn_output_past.copy_(
                            self.model_runner.attn_output_past.index_select(1, refresh_cache_idxs)
                        )
                        self.model_runner.logsumexp.copy_(
                            self.model_runner.logsumexp.index_select(1, refresh_cache_idxs)
                        )
	                        # print(self.tokenizer.decode(output[0][1]))
                if profile:
                    prof.step()

                if use_tqdm:
                    pbar.update(len(output))

                # for seq_id, token_ids in output:
                #     outputs[seq_id] = token_ids
                for seq_id, token_ids, unmask_times,idx in output:
                    outputs[seq_id] = {"token_ids": token_ids, "unmask_times": unmask_times}

        #outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        #results = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs_list]
        outputs_list = [outputs[seq_id] for seq_id in sorted(outputs)]
        results = [
            {
                "text": self._safe_decode(item["token_ids"]),
                "token_ids": self._clean_token_ids(item["token_ids"]),
                "first_unmask_times": item["unmask_times"],
            }
            for item in outputs_list
        ]

        if use_tqdm:
            pbar.close()
        return results
