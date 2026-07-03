#!/usr/bin/env python3
import argparse
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLASH_ROOT = ROOT.parent / "flash-attention"
_cache_root = os.environ.get("TRADO_CACHE_ROOT", f"/dev/shm/flashblock_cache_{os.environ.get('USER', 'user')}")
os.makedirs(_cache_root, exist_ok=True)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(_cache_root, "torch_extensions"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
os.environ.setdefault("XDG_CACHE_HOME", _cache_root)
for path in (ROOT, ROOT / "sample", FLASH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL", str(ROOT / "models/TraDo-8B-Thinking")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mask-token-id", type=int, default=151669)
    args = parser.parse_args()

    from jetengine_ext.llm import LLM

    print(
        "[probe] init "
        f"model={args.model} tp={args.tensor_parallel_size} "
        f"batch={args.batch_size} block={args.block_size} "
        f"gpu_mem_util={args.gpu_memory_utilization}",
        flush=True,
    )
    t0 = time.time()
    llm = LLM(
        args.model,
        enforce_eager=False,
        tensor_parallel_size=args.tensor_parallel_size,
        mask_token_id=args.mask_token_id,
        block_length=args.block_size,
        batch_size=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print("[probe] init_sec", time.time() - t0, flush=True)
    kv = llm.model_runner.kv_cache
    print("kv_cache_shape", tuple(kv.shape), flush=True)
    print("num_kvcache_blocks", llm.config.num_kvcache_blocks, flush=True)
    print("kvcache_block_size", llm.config.kvcache_block_size, flush=True)
    print("capacity_tokens_total_context", llm.config.num_kvcache_blocks * llm.config.kvcache_block_size, flush=True)
    print("kv_dtype", kv.dtype, "kv_device", kv.device, flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
