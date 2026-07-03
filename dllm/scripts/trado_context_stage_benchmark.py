#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import os
import random
import signal
import socket
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.pop("NCCL_BLOCKING_WAIT", None)
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

_cache_root = os.environ.get("TRADO_CACHE_ROOT", f"/dev/shm/flashblock_cache_{os.environ.get('USER', 'user')}")
os.makedirs(_cache_root, exist_ok=True)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(_cache_root, "torch_extensions"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
os.environ.setdefault("XDG_CACHE_HOME", _cache_root)

for p in [REPO, REPO / "sample", REPO / "flash-attention", REPO.parent / "flash-attention"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from jinja2 import Template
from omegaconf import MISSING, OmegaConf


MATH_PROMPT = """<|im_start|>user
Please reason step by step, and put your final answer within \\boxed{}.
{{problem}}<|im_end|>
<|im_start|>assistant
"""

MATH_PROMPT_THINK = """<|im_start|>user
You need to put your final answer in \\boxed{}. This is the problem:
{{problem}}<|im_end|>
<|im_start|>assistant<think>
"""

OPTION_PROMPT = """<|im_start|>user
This is the problem:
{{problem}}
You need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>
<|im_start|>assistant
"""

OPTION_PROMPT_THINK = """<|im_start|>user
This is the problem:
{{problem}}
You need to think step by step and put the final option (A, B, C, or D only—no other character) in \\boxed{}. <|im_end|>
<|im_start|>assistant<think>
"""

CODE_PROMPT_FUNCTION = """<|im_start|>user
{{problem}}
Place your code within a single Python code block ```python ```. Do not include more than one code block. <|im_end|>
<|im_start|>assistant
"""

CODE_PROMPT_STDIO = """<|im_start|>user
This is the problem:
{{problem}}
You should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>
<|im_start|>assistant
"""

CODE_PROMPT_STDIO_THINK = """<|im_start|>user
This is the problem:
{{problem}}
You should put your code in ```python ```. Use input() to read input and print() to produce output in your script. <|im_end|>
<|im_start|>assistant<think>
"""


def find_free_port() -> str:
    s = socket.socket()
    s.bind(("", 0))
    port = str(s.getsockname()[1])
    s.close()
    return port


def mode_to_sparsity(mode: str) -> int:
    if mode == "baseline":
        return 0
    if mode.startswith("s") and mode[1:].isdigit():
        return int(mode[1:])
    raise ValueError(f"Unknown mode: {mode}")


def load_prompts(config, max_prompts: int | None = None) -> list[str]:
    dataset = config.dataset.eval_dataset
    data_dir = Path(os.environ.get("TRADO_DATA_DIR", REPO / "data"))
    with open(data_dir / f"{dataset}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    num_node = int(config.experiment.get("num_node", 1))
    node_index = int(config.experiment.get("node_index", 0))
    if num_node > 1:
        total = len(data)
        start = (total * node_index) // num_node
        end = (total * (node_index + 1)) // num_node
        data = data[start:end]

    start = int(config.dataset.get("start", 0))
    limit = int(config.dataset.get("limit", -1))
    if start > 0:
        data = data[start:]
    if limit > 0:
        data = data[:limit]
    if max_prompts is not None and max_prompts > 0:
        data = data[:max_prompts]

    data_type = config.dataset.data_type
    if data_type == "option":
        template = OPTION_PROMPT_THINK if config.rollout.start_with_think else OPTION_PROMPT
    elif data_type == "code":
        template = None
    else:
        template = MATH_PROMPT_THINK if config.rollout.start_with_think else MATH_PROMPT

    k_sample = int(config.rollout.get("num_response_per_task", 1))
    prompts = []
    for item in data:
        if data_type == "code":
            if item.get("test_method") == "stdio":
                item_template = CODE_PROMPT_STDIO_THINK if config.rollout.start_with_think else CODE_PROMPT_STDIO
            else:
                item_template = CODE_PROMPT_FUNCTION + item.get("prefix", "")
            prompt = Template(item_template).render(problem=item["question"])
        else:
            prompt = Template(template).render(problem=item["question"])
        prompts.extend([prompt] * k_sample)
    return prompts


def stop_token_ids(tokenizer, stop_token_list):
    if not stop_token_list:
        return []
    ids, seen = [], set()
    for s in stop_token_list:
        if isinstance(s, int):
            tokenized = [s]
        elif isinstance(s, str):
            tokenized = tokenizer.encode(s, add_special_tokens=False)
        elif isinstance(s, (list, tuple)) and all(isinstance(x, int) for x in s):
            tokenized = list(s)
        else:
            continue
        if len(tokenized) == 1 and tokenized[0] not in seen:
            seen.add(tokenized[0])
            ids.append(tokenized[0])
    return ids


def summarize_window(label: str, xs: list[dict]) -> dict:
    lats = [x["latency_s"] for x in xs]
    toks = [x["num_processed"] for x in xs]
    summary = {
        "stage": label,
        "steps": len(xs),
        "tokens": sum(toks),
        "avg_latency_ms": statistics.mean(lats) * 1000.0,
        "median_latency_ms": statistics.median(lats) * 1000.0,
        "min_latency_ms": min(lats) * 1000.0,
        "max_latency_ms": max(lats) * 1000.0,
        "tps": sum(toks) / sum(lats) if sum(lats) > 0 else 0.0,
        "avg_cur_tps": statistics.mean(x["current_tps"] for x in xs),
        "avg_context": statistics.mean(x["total_context"] for x in xs),
        "avg_n": statistics.mean(x["n"] for x in xs),
        "avg_schedule_time_ms": statistics.mean(x["schedule_time_s"] for x in xs) * 1000.0,
        "avg_run_time_ms": statistics.mean(x["run_time_s"] for x in xs) * 1000.0,
        "avg_write_time_ms": statistics.mean(x["write_time_s"] for x in xs) * 1000.0,
        "avg_post_schedule_ms": statistics.mean(x["post_schedule_time_s"] for x in xs) * 1000.0,
        "avg_read_time_ms": statistics.mean(x["read_time_s"] for x in xs) * 1000.0,
        "avg_update_kv_cache": statistics.mean(x["update_kv_cache"] for x in xs),
        "avg_no_update_kv_cache": statistics.mean(x["no_update_kv_cache"] for x in xs),
    }
    optional_ms_fields = [
        "prepare_ms",
        "forward_sync_ms",
        "attention_ms",
        "non_attention_forward_ms",
        "non_attention_local_ms",
        "detail_embed_ms",
        "detail_input_norm_ms",
        "detail_qkv_norm_rope_ms",
        "detail_block_attn_call_ms",
        "detail_o_proj_ms",
        "detail_self_attn_total_ms",
        "detail_post_attn_norm_ms",
        "detail_mlp_ms",
        "detail_decoder_layer_total_ms",
        "detail_final_norm_ms",
        "detail_lm_head_ms",
        "post_sync_ms",
        "return_bookkeeping_ms",
        "post_inner_logits_processor_ms",
        "post_inner_sample_gather_ms",
        "post_inner_block_tensor_ms",
        "post_inner_batched_transfer_ms",
        "post_inner_batched_transfer_count",
        "post_inner_loop_update_ms",
        "post_inner_filter_deallocate_ms",
        "post_inner_total_profiled_ms",
    ]
    for field in optional_ms_fields:
        if field in xs[0]:
            summary[f"avg_{field}"] = statistics.mean(float(x.get(field, 0.0)) for x in xs)
    return summary


def summarize_rows(rows: list[dict], checkpoints: list[int], window: int) -> list[dict]:
    summaries = []
    for checkpoint in checkpoints:
        lo = checkpoint
        hi = checkpoint + window
        label = f"{lo//1000}k-{hi//1000}k"
        xs = [
            r for r in rows
            if r["num_processed"] > 0 and lo <= r["total_context"] < hi
        ]
        if xs:
            summaries.append(summarize_window(label, xs))
        else:
            summaries.append({
                "stage": label,
                "steps": 0,
                "tokens": 0,
                "avg_latency_ms": 0.0,
                "median_latency_ms": 0.0,
                "min_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "tps": 0.0,
                "avg_cur_tps": 0.0,
                "avg_context": 0.0,
                "avg_n": 0.0,
                "avg_post_schedule_ms": 0.0,
                "avg_read_time_ms": 0.0,
            })
    return summaries


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_time_logs(path: Path, rows: list[dict]):
    if os.environ.get("TRADO_NO_TIME_LOGS", "0") == "1":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)


def patch_dist_port(port: str):
    import torch.distributed as dist
    real_init = dist.init_process_group

    def wrapped(backend, init_method=None, *args, **kwargs):
        if isinstance(init_method, str) and init_method.startswith("tcp://localhost:2333"):
            init_method = f"tcp://127.0.0.1:{port}"
        return real_init(backend, init_method, *args, **kwargs)

    dist.init_process_group = wrapped


def run_mode(args, mode: str) -> dict:
    port = args.master_port or find_free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["JE_TCP_PORT"] = port
    patch_dist_port(port)

    from transformers import AutoTokenizer
    from jetengine_ext.llm import LLM
    from jetengine_ext.sampling_params import SamplingParams
    from jetengine_ext.engine.llm_engine import move_to_end_torch

    config = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(args.overrides or [])
    config = OmegaConf.merge(config, cli)

    config.model = args.model or config.model
    config.rollout.tensor_parallel_size = args.tensor_parallel_size
    config.rollout.max_active = args.max_active
    config.rollout.gpu_memory_utilization = args.gpu_memory_utilization
    config.rollout.sparsity_ratio = mode_to_sparsity(mode)
    if args.dynamic_threshold is not None:
        config.rollout.dynamic_threshold = args.dynamic_threshold
    if args.max_tokens is not None:
        config.rollout.max_token = args.max_tokens

    prompts = load_prompts(config, args.max_prompts)
    if args.shuffle:
        random.Random(args.seed).shuffle(prompts)
    if len(prompts) < args.max_active:
        raise RuntimeError(f"Need at least max_active prompts, got {len(prompts)}")

    model_path = os.path.expanduser(config.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if OmegaConf.select(config, "rollout.stop_token_list", default=MISSING) is not MISSING:
        stop_words = stop_token_ids(tokenizer, config.rollout.stop_token_list)
    else:
        stop_words = []

    sampling_params = SamplingParams(
        temperature=config.rollout.temperature,
        topk=config.rollout.top_k,
        topp=config.rollout.top_p,
        max_tokens=config.rollout.max_token,
        remasking_strategy=config.rollout.remasking_strategy,
        block_length=config.rollout.block_size,
        denoising_steps=config.rollout.denoising_steps_per_block,
        dynamic_threshold=config.rollout.dynamic_threshold,
        stop_words=stop_words,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_step_path = out_dir / f"{mode}_steps.csv"
    summary_path = out_dir / f"{mode}_summary.csv"
    live_summary_path = out_dir / f"{mode}_summary_live.csv"
    json_path = out_dir / f"{mode}_summary.json"
    time_logs_path = Path(os.environ.get("JETENGINE_TIME_LOG_PATH", out_dir / f"{mode}_time_logs.pt"))

    llm = None
    rows: list[dict] = []
    live_summaries: list[dict] = []
    checkpoints: list[int] = []
    printed_checkpoints: set[int] = set()
    max_context_seen = 0
    total_generated = 0
    pending_idx = 0
    wall_start = time.perf_counter()

    print(
        f"[start] mode={mode} prompts={len(prompts)} max_active={args.max_active} "
        f"stop_context={args.stop_context} tp={args.tensor_parallel_size} port={port}",
        flush=True,
    )

    try:
        llm = LLM(
            model_path,
            enforce_eager=True if os.environ.get("TRADO_SYNC_PROFILE", "0") == "1" else (False if args.tensor_parallel_size > 1 else True),
            tensor_parallel_size=args.tensor_parallel_size,
            mask_token_id=args.mask_token_id,
            block_length=config.rollout.block_size,
            batch_size=args.max_active,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        kv_capacity = int(llm.config.num_kvcache_blocks * llm.config.kvcache_block_size)
        if args.checkpoint_end < 0:
            args.checkpoint_end = (
                (kv_capacity - args.checkpoint_window)
                // args.checkpoint_interval
                * args.checkpoint_interval
            )
            if args.checkpoint_end < args.checkpoint_start:
                args.checkpoint_end = args.checkpoint_start
        if args.stop_context < 0 or args.stop_context < args.checkpoint_end + args.checkpoint_window:
            args.stop_context = args.checkpoint_end + args.checkpoint_window
        checkpoints = list(range(args.checkpoint_start, args.checkpoint_end + 1, args.checkpoint_interval))
        print(
            f"[kv] mode={mode} capacity_tokens_total_context={kv_capacity} "
            f"checkpoint_end={args.checkpoint_end} stop_context={args.stop_context}",
            flush=True,
        )
        llm.scheduler.consistent_sampling_params = os.environ.get(
            "TRADO_CONSISTENT_SAMPLING_PARAMS", "1"
        ) != "0"

        initial = min(args.max_active, len(prompts))
        for i in range(initial):
            llm.add_request(prompts[i], sampling_params, config.rollout.sparsity_ratio)
        pending_idx = initial

        step_idx = 0
        while (not llm.is_finished() or pending_idx < len(prompts)):
            running = getattr(llm.scheduler, "running", [])
            deficit = args.max_active - len(running)
            while deficit > 0 and pending_idx < len(prompts):
                llm.add_request(
                    prompts[pending_idx],
                    sampling_params,
                    config.rollout.sparsity_ratio,
                )
                pending_idx += 1
                deficit -= 1

            st = time.time()
            step_result = llm.step()
            latency = time.time() - st
            step_idx += 1
            profile_info = {}
            if len(step_result) == 11:
                (
                    output,
                    num_processed,
                    update_kv_cache,
                    no_update_kv_cache,
                    total_context,
                    run_time,
                    write_time,
                    n,
                    schedule_time,
                    post_schedule_time,
                    profile_info,
                ) = step_result
            elif len(step_result) == 10:
                (
                    output,
                    num_processed,
                    update_kv_cache,
                    no_update_kv_cache,
                    total_context,
                    run_time,
                    write_time,
                    n,
                    schedule_time,
                    post_schedule_time,
                ) = step_result
            else:
                output, num_processed, update_kv_cache, no_update_kv_cache = step_result
                total_context = run_time = write_time = n = schedule_time = post_schedule_time = 0

            if config.rollout.sparsity_ratio > 0 and len(output) > 0:
                refresh_cache_idxs = move_to_end_torch(args.max_active, [x[3] for x in output])
                refresh_cache_idxs = refresh_cache_idxs.to(llm.model_runner.attn_output_past.device)
                llm.model_runner.attn_output_past.copy_(llm.model_runner.attn_output_past.index_select(1, refresh_cache_idxs))
                llm.model_runner.logsumexp.copy_(llm.model_runner.logsumexp.index_select(1, refresh_cache_idxs))
            total_generated += num_processed
            max_context_seen = max(max_context_seen, int(total_context))
            current_tps = num_processed / latency if latency > 0 else 0.0
            read_time = llm.model_runner.get_read_time()
            row = {
                "step": step_idx,
                "mode": mode,
                "latency_s": latency,
                "latency_ms": latency * 1000.0,
                "num_processed": int(num_processed),
                "total_context": int(total_context),
                "max_context_seen": max_context_seen,
                "pending_idx": pending_idx,
                "running": len(getattr(llm.scheduler, "running", [])),
                "run_time_s": float(run_time),
                "write_time_s": float(write_time),
                "schedule_time_s": float(schedule_time),
                "post_schedule_time_s": float(post_schedule_time),
                "read_time_s": float(read_time),
                "n": int(n),
                "current_tps": current_tps,
                "update_kv_cache": int(update_kv_cache),
                "no_update_kv_cache": int(no_update_kv_cache),
            }
            if profile_info:
                row.update({
                    "prepare_ms": float(profile_info.get("prepare_ms", 0.0)),
                    "forward_sync_ms": float(profile_info.get("forward_sync_ms", 0.0)),
                    "attention_ms": float(profile_info.get("attention_ms", 0.0)),
                    "non_attention_forward_ms": float(profile_info.get("non_attention_forward_ms", 0.0)),
                    "post_sync_ms": float(post_schedule_time) * 1000.0,
                })
                for key, value in profile_info.items():
                    if key.startswith("post_inner_") or key.endswith("_ms"):
                        row[key] = float(value)
            rows.append(row)

            for checkpoint in checkpoints:
                if checkpoint in printed_checkpoints:
                    continue
                window_end = checkpoint + args.checkpoint_window
                if max_context_seen < window_end:
                    continue
                recent = [
                    r for r in rows
                    if r["num_processed"] > 0 and checkpoint <= r["total_context"] < window_end
                ]
                if recent:
                    window_summary = summarize_window(
                        f"{checkpoint//1000}k-{window_end//1000}k",
                        recent,
                    )
                    live_summaries.append(window_summary)
                    write_csv(live_summary_path, live_summaries)
                    save_time_logs(time_logs_path, rows)
                    profile_bits = []
                    for label, key in [
                        ("prepare", "avg_prepare_ms"),
                        ("forward", "avg_forward_sync_ms"),
                        ("attn", "avg_attention_ms"),
                        ("nonattn", "avg_non_attention_forward_ms"),
                        ("schedule", "avg_schedule_time_ms"),
                        ("post_sync", "avg_post_sync_ms"),
                        ("post_inner", "avg_post_inner_total_profiled_ms"),
                    ]:
                        if key in window_summary:
                            profile_bits.append(f"{label}_ms={window_summary[key]:.2f}")
                    profile_bits.extend([
                        f"write_ms={window_summary['avg_write_time_ms']:.2f}",
                        f"read_ms={window_summary['avg_read_time_ms']:.2f}",
                        f"n_mb={window_summary['avg_n'] / 1e6:.2f}",
                    ])
                    print(
                        f"[stage] mode={mode} {window_summary['stage']} "
                        f"steps={window_summary['steps']} "
                        f"avg_ms={window_summary['avg_latency_ms']:.2f} "
                        f"median_ms={window_summary['median_latency_ms']:.2f} "
                        f"tps={window_summary['tps']:.2f} "
                        f"{' '.join(profile_bits)} "
                        f"max_context={max_context_seen}",
                        flush=True,
                    )
                else:
                    print(
                        f"[stage] mode={mode} {checkpoint//1000}k-{window_end//1000}k no decode rows",
                        flush=True,
                    )
                printed_checkpoints.add(checkpoint)

            if step_idx % args.log_every == 0 or max_context_seen >= args.stop_context:
                print(
                    f"[step] mode={mode} step={step_idx} context={total_context} "
                    f"max_context={max_context_seen} latency_ms={latency*1000:.2f} "
                    f"num_processed={num_processed} cur_tps={current_tps:.2f} pending={pending_idx}",
                    flush=True,
                )

            if max_context_seen >= args.stop_context:
                print(f"[stop] mode={mode} reached max_context={max_context_seen}", flush=True)
                break

        summaries = summarize_rows(rows, checkpoints, args.checkpoint_window)
        write_csv(per_step_path, rows)
        write_csv(summary_path, summaries)
        save_time_logs(time_logs_path, rows)
        result = {
            "mode": mode,
            "config": args.config,
            "model": model_path,
            "output_dir": str(out_dir),
            "steps_csv": str(per_step_path),
            "summary_csv": str(summary_path),
            "time_logs_pt": str(time_logs_path),
            "stop_context": args.stop_context,
            "max_context_seen": max_context_seen,
            "total_generated_tokens": total_generated,
            "wall_s": time.perf_counter() - wall_start,
            "stage_summaries": summaries,
            "checkpoint_start": args.checkpoint_start,
            "checkpoint_end": args.checkpoint_end,
            "checkpoint_interval": args.checkpoint_interval,
            "checkpoint_window": args.checkpoint_window,
        }
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("[result]", json.dumps(result, sort_keys=True), flush=True)
        return result
    finally:
        if llm is not None:
            try:
                llm.exit()
            except Exception:
                pass
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/trado_eval_sparsity.yaml"))
    ap.add_argument("--model", default=os.environ.get("MODEL", str(REPO / "models/TraDo-8B-Thinking")))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", default="baseline", help="baseline, s2, s3, or s4")
    ap.add_argument("--max-active", type=int, default=128)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--mask-token-id", type=int, default=151669)
    ap.add_argument("--dynamic-threshold", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--stop-context", type=int, default=800_000)
    ap.add_argument("--stage-interval", type=int, default=100_000, help="Deprecated alias; kept for launcher compatibility.")
    ap.add_argument("--checkpoint-start", type=int, default=100_000)
    ap.add_argument("--checkpoint-end", type=int, default=800_000)
    ap.add_argument("--checkpoint-interval", type=int, default=100_000)
    ap.add_argument("--checkpoint-window", type=int, default=10_000)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--master-port", default=None)
    ap.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides, e.g. dataset.eval_dataset=MATH500")
    args = ap.parse_args()
    if args.checkpoint_end >= 0 and args.stop_context >= 0 and args.stop_context < args.checkpoint_end + args.checkpoint_window:
        args.stop_context = args.checkpoint_end + args.checkpoint_window

    def handle_signal(sig, _frame):
        print(f"[signal] got {sig}, exiting", flush=True)
        sys.exit(128 + sig)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    result = run_mode(args, args.mode)
    print(f"[done] mode={args.mode} summary={result['summary_csv']}", flush=True)


if __name__ == "__main__":
    main()
