# P07 — Fused RMSNorm Triton kernel

Kernel intuition is what separates infra engineers from API callers: RMSNorm
is memory-bound, so the win is entirely about DRAM traffic, not FLOPs.

## Why fusion wins (the one paragraph)

`torch_rmsnorm` launches 4 kernels: `pow` (read x, write t1), `mean` (read t1),
`rsqrt` (scalar), `mul` (read x, t2, w — write y). The row streams through
DRAM ~3×. The Triton kernel loads each element **once** into SRAM, reduces in
registers/shared, and stores once. Same FLOPs, ~1/3 the bytes → expect
**1.5–3×** on A10G at hidden ≥ 3584, less at small dims (launch overhead).

## Kernel design notes

- One program per row (`program_id(0)`), `BLOCK = next_pow2(hidden)` threads.
- `mask=cols < N` handles non-power-of-2 dims (3584 → BLOCK 4096, 12.5% idle
  lanes — note this in results; padding waste is part of the honest story).
- Accumulate in **fp32** (`to(tl.float32)` before square) — fp16 accumulation
  over 3584 elements loses the small variances; the classic silent bug.
- `tl.int64` program id: rows × stride can overflow int32 address math.

## Files

| File | Runs where | Purpose |
|---|---|---|
| `rmsnorm_triton.py` | GPU (`--bench`) | kernel + correctness + `do_bench` over rows × {3584, 4096} |
| `test_correctness.py` | Mac CPU ✅ | reference properties, guard behavior, fp16 floor |

## GPU procedure, one boot (~30 min, ~$0.40)

On any CUDA host with torch+triton (`pip install torch triton`):
`python3 rmsnorm_triton.py --bench`. Paste the table:

## Results — A10G, vLLM image torch/triton (2026-09-20)

Correctness: max_abs=1.95e-03, max_rel=9.61e-04 (beats the 1e-2 bar and the
2.2e-3 fp16 floor — fp32 accumulation justified).

| rows | dim | torch_ms | triton_ms | speedup | GB/s |
|---|---|---|---|---|---|
| 512 | 3584 | 0.064 | 0.018 | 3.59x | 619 |
| 512 | 4096 | 0.076 | 0.021 | 3.70x | 613 |
| 2048 | 3584 | 0.231 | 0.065 | 3.57x | 682 |
| 2048 | 4096 | 0.261 | 0.073 | 3.56x | 685 |
| 8192 | 3584 | 0.878 | 0.246 | 3.57x | 716 |
| 8192 | 4096 | 0.998 | 0.281 | 3.55x | 715 |

Caveat (kept): GB/s exceeds A10G's ~600 GB/s HBM ceiling because `do_bench`
repeats identical tensors (L2 reuse). The 3.5× ratio is the honest number —
same caching applies to both sides. Raw: `runs/20260920-p7/`.
