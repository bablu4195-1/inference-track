#!/usr/bin/env python3
"""P7 CPU-runnable tests: reference math properties + triton guard behavior.

Runs on the Mac with plain torch CPU (no CUDA, no triton needed):
  1. Reference RMSNorm normalizes rows: RMS(out / w) ~= 1
  2. Reference matches the closed-form definition elementwise
  3. triton_rmsnorm raises a clear error (not an ImportError traceback) off-CUDA
  4. fp32 vs fp16 reference agreement bounds the precision floor for the GPU run

Usage: python3 p07-triton-kernel/test_correctness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rmsnorm_triton import HAS_TRITON, torch_rmsnorm, triton_rmsnorm  # noqa: E402

torch.manual_seed(7)


def test_row_normalization() -> None:
    x = torch.randn(64, 512)
    w = torch.randn(512)
    y = torch_rmsnorm(x, w)
    rms = (y / w).pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5), rms.max()
    print("PASS row normalization: RMS(y/w) == 1")


def test_closed_form() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    w = torch.tensor([1.0, 1.0, 1.0, 1.0])
    expect = x / ((1 + 4 + 9 + 16) / 4 + 1e-6) ** 0.5
    assert torch.allclose(torch_rmsnorm(x, w), expect, atol=1e-6)
    print("PASS closed form on [1,2,3,4]")


def test_weight_scaling() -> None:
    x = torch.randn(32, 256)
    assert torch.allclose(torch_rmsnorm(x, torch.ones(256)),
                          torch_rmsnorm(x, 2 * torch.ones(256)) / 2, atol=1e-6)
    print("PASS output linear in weight")


def test_triton_guard() -> None:
    if HAS_TRITON and torch.cuda.is_available():
        print("SKIP guard test (CUDA present — real check runs on GPU)")
        return
    try:
        triton_rmsnorm(torch.randn(4, 8), torch.randn(8))
    except (RuntimeError, AssertionError) as e:
        print(f"PASS triton guard raises cleanly: {type(e).__name__}")
        return
    raise SystemExit("FAIL: triton path should refuse off-CUDA")


def test_precision_floor() -> None:
    x = torch.randn(256, 1024)
    w = torch.randn(1024)
    ref32 = torch_rmsnorm(x, w)
    ref16 = torch_rmsnorm(x.half(), w.half()).float()
    rel = ((ref32 - ref16).abs() / ref32.abs().clamp_min(1e-3)).max().item()
    print(f"INFO fp16-vs-fp32 reference max_rel={rel:.2e} "
          f"(GPU kernel must beat ~1e-2; floor is ~{rel:.0e})")


def main() -> None:
    test_row_normalization()
    test_closed_form()
    test_weight_scaling()
    test_triton_guard()
    test_precision_floor()
    print("ALL CPU TESTS PASS")


if __name__ == "__main__":
    main()
