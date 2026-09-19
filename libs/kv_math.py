"""KV-cache VRAM predictor (P3). Validated against live /metrics in P3.

Math: per-token KV bytes = 2 (K+V) * layers * kv_heads * head_dim * bytes/elem.
GQA-aware via num_key_value_heads. Add ~10% block-table overhead for
paged blocks (block_size 16 default).

Usage:
  python3 kv_math.py --model qwen2.5-7b --seq-len 8192 --batch 4
  python3 kv_math.py --table          # max batch per seq-len on A10G 24GB
"""

from __future__ import annotations

import argparse

# Hidden arch facts, verified against HF configs.
MODEL_SPECS = {
    "qwen2.5-7b": {
        "params_b": 7.6, "layers": 28, "hidden": 3584,
        "q_heads": 28, "kv_heads": 4, "head_dim": 128,
        "weights_fp16_gb": 15.2, "max_model_len": 32768,
    },
    "llama-3.1-8b": {
        "params_b": 8.0, "layers": 32, "hidden": 4096,
        "q_heads": 32, "kv_heads": 8, "head_dim": 128,
        "weights_fp16_gb": 16.1, "max_model_len": 131072,
    },
    "tinyllama-1.1b": {
        "params_b": 1.1, "layers": 22, "hidden": 2048,
        "q_heads": 32, "kv_heads": 4, "head_dim": 64,
        "weights_fp16_gb": 2.2, "max_model_len": 2048,
    },
}

A10G_GB = 24.0
GPU_UTIL = 0.9            # --gpu-memory-utilization
USABLE_GB = A10G_GB * GPU_UTIL
BLOCK_OVERHEAD = 1.10     # paged-block waste factor
DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1}
# Weight footprint divisor vs FP16; KV stays FP16 unless the dtype says otherwise.
WEIGHTS_DIV = {"fp16": 1, "bf16": 1, "fp8": 2, "awq": 4}
KV_DTYPE = {"fp16": "fp16", "bf16": "bf16", "fp8": "fp8", "awq": "fp16"}


def kv_bytes_per_token(spec: dict, dtype: str = "fp16") -> int:
    kv_dt = KV_DTYPE.get(dtype, dtype)  # awq weights, fp16 KV
    return 2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"] * DTYPE_BYTES[kv_dt]


def kv_gb(seq_len: int, batch: int, model: str = "qwen2.5-7b",
           dtype: str = "fp16") -> float:
    spec = MODEL_SPECS[model]
    raw = kv_bytes_per_token(spec, dtype) * seq_len * batch
    return raw * BLOCK_OVERHEAD / 1e9


def total_gb(seq_len: int, batch: int, model: str = "qwen2.5-7b",
             dtype: str = "fp16", weights_gb: float | None = None) -> float:
    spec = MODEL_SPECS[model]
    w = weights_gb if weights_gb is not None else spec["weights_fp16_gb"]
    w = w / WEIGHTS_DIV.get(dtype, 1)
    return w + 1.0 + kv_gb(seq_len, batch, model, KV_DTYPE.get(dtype, "fp16"))


def fits(seq_len: int, batch: int, **kw) -> bool:
    return total_gb(seq_len, batch, **kw) <= USABLE_GB


def max_batch(seq_len: int, **kw) -> int:
    b = 0
    while fits(seq_len, b + 1, **kw) and b < 512:
        b += 1
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description="KV-cache VRAM predictor")
    ap.add_argument("--model", default="qwen2.5-7b", choices=list(MODEL_SPECS))
    ap.add_argument("--seq-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp8", "awq"])
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()

    spec = MODEL_SPECS[a.model]
    print(f"model={a.model} kv/token={kv_bytes_per_token(spec, a.dtype)/1024:.1f}KiB "
          f"usable={USABLE_GB:.1f}GB (A10G x {GPU_UTIL})")
    if a.table:
        print(f"{'seq_len':>8} {'max_batch':>9} {'kv_GB@max':>10}")
        for s in (1024, 2048, 4096, 8192, 16384, 32768):
            if s > spec["max_model_len"]:
                continue
            b = max_batch(s, model=a.model, dtype=a.dtype)
            print(f"{s:>8} {b:>9} {kv_gb(s, b, a.model, a.dtype):>10.2f}")
        return
    kv = kv_gb(a.seq_len, a.batch, a.model, a.dtype)
    tot = total_gb(a.seq_len, a.batch, a.model, a.dtype)
    print(f"seq={a.seq_len} batch={a.batch} dtype={a.dtype}: "
          f"KV={kv:.2f}GB total~{tot:.2f}GB -> {'FITS' if tot <= USABLE_GB else 'OOM (predicted)'}")


if __name__ == "__main__":
    main()
