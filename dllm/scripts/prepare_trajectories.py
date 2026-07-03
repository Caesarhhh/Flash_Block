#!/usr/bin/env python
import argparse
import os
import random
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate dense rollout trajectories used by FlashBlock dLLM training."
    )
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default="data/train_trajectories.pt")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--dynamic-threshold", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--remasking-strategy",
        default="low_confidence_dynamic",
        choices=["sequential", "low_confidence_static", "low_confidence_dynamic", "entropy_bounded"],
    )
    parser.add_argument("--seed", type=int, default=10085)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_seed(seed: int):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt_texts(tokenizer, data):
    prompts = []
    for item in data:
        question = "Please reason step by step, and put your final answer within \\boxed{}.\n" + item["question"]
        messages = [{"role": "user", "content": question}]
        input_ids = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).input_ids[0]
        prompts.append(tokenizer.decode(input_ids))
    return prompts


def main():
    args = parse_args()

    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm
    from transformers import AutoTokenizer

    from sample.jetengine_ext.llm import LLM
    from sample.jetengine_ext.sampling_params import SamplingParams
    from utils.data import read_bs

    set_seed(args.seed)

    config = OmegaConf.load(args.config)
    model_path = args.model or os.environ.get("MODEL") or config.paths.model
    model_path = os.path.expanduser(str(model_path))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. "
            "Download the model locally or pass --model/ MODEL=/path/to/TraDo-8B-Thinking."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data = read_bs(config)
    if args.limit and args.limit > 0:
        data = data[: args.limit]
    prompts = build_prompt_texts(tokenizer, data)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories = []
    if args.resume and output_path.exists():
        trajectories = torch.load(output_path, map_location="cpu")
        if not isinstance(trajectories, list):
            raise TypeError(f"Expected a list in {output_path}, got {type(trajectories)!r}")
        prompts = prompts[len(trajectories) :]

    if not prompts:
        print(f"[trajectory] nothing to generate; existing={len(trajectories)} output={output_path}")
        return

    sampling_params = SamplingParams(
        temperature=args.temperature,
        topk=args.top_k,
        topp=args.top_p,
        max_tokens=args.max_tokens,
        remasking_strategy=args.remasking_strategy,
        block_length=args.block_size,
        denoising_steps=args.denoising_steps,
        dynamic_threshold=args.dynamic_threshold,
        stop_words=[],
    )

    llm = LLM(
        model_path,
        enforce_eager=args.tensor_parallel_size <= 1,
        tensor_parallel_size=args.tensor_parallel_size,
        mask_token_id=151669,
        block_length=args.block_size,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    try:
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Trajectories", dynamic_ncols=True):
            batch_prompts = prompts[start : start + args.batch_size]
            outputs = llm.generate_streaming(
                batch_prompts,
                sampling_params,
                max_active=args.batch_size,
                use_tqdm=False,
                sparsity_ratio=0,
            )
            trajectories.extend(
                prompt + output["text"]
                for prompt, output in zip(batch_prompts, outputs)
            )
            torch.save(trajectories, output_path)
    finally:
        try:
            llm.exit()
        except Exception:
            pass

    print(f"[trajectory] saved {len(trajectories)} trajectories to {output_path}")


if __name__ == "__main__":
    main()
