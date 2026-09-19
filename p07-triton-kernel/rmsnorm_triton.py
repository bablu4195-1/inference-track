"""P7 fused RMSNorm Triton kernel vs PyTorch baseline.

RMSNorm (used by Llama/Qwen instead of LayerNorm):
    y = x / sqrt(mean(x^2) + eps) * w            (no mean-centering, no bias)

Why fusion wins: torch reference launches 4+ kernels (square, sum, rsqrt,
scale) each streaming the row through DRAM. The fused kernel loads each
element ONCE. RMSNorm is memory-bound -> DRAM traffic is the whole game.

Run:
  GPU:  python3 p07-triton-kernel/rmsnorm_triton.py --bench   # correctness + timings
  Mac:  python3 p07-triton-kernel/test_correctness.py         # reference math only
"""

from __future__ import annotations

import argparse
import sys

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception as e:  # Mac CPU / no CUDA: kernel path unavailable, ref still works
    HAS_TRITON = False
    _TRITON_ERR = e


def torch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd(X, Y, W, stride_x, stride_y, N, eps,
                     BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        X += row * stride_x
        Y += row * stride_y
        cols = tl.arange(0, BLOCK)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=cols < N)
        var = tl.sum(x * x, 0) / N
        rstd = 1 / tl.sqrt(var + eps)
        # fp32 accumulate, implicit downcast on store to Y (same dtype as X)
        tl.store(Y + cols, x * rstd * w.to(tl.float32), mask=cols < N)

    def triton_rmsnorm(x: torch.Tensor, w: torch.Tensor,
                       eps: float = 1e-6) -> torch.Tensor:
        assert x.is_cuda, "triton kernel needs CUDA (this is the P7 GPU step)"
        y = torch.empty_like(x)
        N = x.shape[-1]
        _rmsnorm_fwd[(x.shape[0],)](x, y, w, x.stride(0), y.stride(0),
                                    N, eps, BLOCK=triton.next_power_of_2(N))
        return y
else:
    def triton_rmsnorm(x, w, eps=1e-6):  # type: ignore
        raise RuntimeError(f"triton unavailable on this host: {_TRITON_ERR}")


def check(rows: int = 512, dim: int = 3584, dtype=torch.float16,
          device: str = "cuda") -> dict:
    """Correctness: triton vs torch. Returns max abs/rel error."""
    dev = torch.device(device)
    torch.manual_seed(0)
    x = torch.randn(rows, dim, dtype=dtype, device=dev)
    w = torch.randn(dim, dtype=dtype, device=dev)
    ref = torch_rmsnorm(x.float(), w.float()).to(dtype)
    got = triton_rmsnorm(x, w)
    return {"max_abs": (got - ref).abs().max().item(),
            "max_rel": ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()}


def bench_fn(fn, *args, rep: int = 200) -> float:
    import triton.testing
    return triton.testing.do_bench(lambda: fn(*args), rep=rep)  # ms


def run_bench() -> None:
    assert torch.cuda.is_available(), "bench needs CUDA"
    print(f"{'rows':>6} {'dim':>5} {'torch_ms':>9} {'triton_ms':>10} {'speedup':>8} {'GB/s_triton':>11}")
    for rows in (512, 2048, 8192):
        for dim in (3584, 4096):  # Qwen2.5-7B / Llama-8B hidden sizes
            x = torch.randn(rows, dim, dtype=torch.float16, device="cuda")
            w = torch.randn(dim, dtype=torch.float16, device="cuda")
            t_torch = bench_fn(torch_rmsnorm, x, w)
            t_tri = bench_fn(triton_rmsnorm, x, w)
            gbs = rows * dim * 2 * 3 / (t_tri / 1e3) / 1e9  # read x + read w + write y
            print(f"{rows:>6} {dim:>5} {t_torch:>9.3f} {t_tri:>10.3f} "
                  f"{t_torch / t_tri:>7.2f}x {gbs:>11.0f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="P7 RMSNorm triton vs torch")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--dim", type=int, default=3584)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("No CUDA here (Mac). Run test_correctness.py instead; "
                 "--bench is the $0.40 GPU step.")
    r = check(a.rows, a.dim)
    print(f"correctness: max_abs={r['max_abs']:.2e} max_rel={r['max_rel']:.2e}")
    assert r["max_rel"] < 1e-2, "kernel disagrees with reference"
    if a.bench:
        run_bench()


if __name__ == "__main__":
    main()
