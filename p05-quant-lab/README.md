# P05 — Quantization comparison lab

Quantization decisions need measured tradeoffs, not vibes: same model, three
precisions, one workload, one quality probe.

## Matrix

| config | weights | serve flags | Expectation |
|---|---|---|---|
| fp16 | ~15 GB | (baseline) | reference quality, most VRAM |
| fp8 | ~8 GB + fp8 KV | `--quantization fp8 --kv-cache-dtype fp8` | near-lossless, more headroom |
| awq | ~4 GB INT4 | `--quantization awq`, `-AWQ` repo | smallest, small quality tax |

Dropped 4th config (GPTQ-INT8) per $128 envelope — AWQ covers the INT4 story.

## Quality method (honest scoping)

Perplexity over `sample.txt` (~600 original words, 5 genres) computed from
`/v1/completions` **prompt logprobs** — no extra deps, runs from the Mac
against any config. Relative-only: any sample bias cancels across configs
(same text, same chunking). This is a smoke signal, not a benchmark — a
>5% ppl regression flags real damage; fine differences need lm-eval (stretch).

## GPU procedure, one boot (~1.5h, ~$0.80)

`./p05-quant-lab/serve-matrix.sh` → per config: VRAM-at-rest (`nvidia-smi`),
P2 quick bench, perplexity probe → `compare.py --report` table below.

## Result table — first GPU run 2026-09-19/20 (Qwen2.5-7B, A10G, P2 quick)

| config | VRAM MiB | TTFT p50 | ITL95 p50 | tok/s | ppl | ppl Δ% |
|---|---|---|---|---|---|---|
| fp16 | 20075 | 510 | 52.6 | 57 | 21.37 | — |
| fp8-kv | 20095 | 451 | 53.0 | 59 | 108.15 | +406% ❌ |
| awq | 19679 | 463 | 29.2 | 119 | 21.85 | +2.2% ✅ |

Findings (all kept, including the failures):
- **Runtime FP8 weights DO NOT RUN on A10G here**: inductor `auto_functionalized`
  assert, then Cutlass `sm80_epilogue` hard-fail with `--enforce-eager`.
  Config pivoted to FP8-KV-only. Needs newer vLLM/CUDA or Hopper — documented,
  not hidden.
- **Uncalibrated FP8-KV destroys quality** (ppl 21→108, +406%): the server logs
  warned (`KV cache scaling factor 1.0 ... may cause accuracy issues`) and it
  was right. FP8-KV needs calibrated scales to be viable.
- **AWQ is the clear winner**: +2.2% ppl (noise-adjacent on 437 tokens), ~2×
  throughput, ITL nearly halved. Rest-VRAM barely differs (pool preallocation
  dominates) — the win is KV *headroom*, exactly as `kv_math.py` predicted.
Raw: `runs/20260919-p5/`.
