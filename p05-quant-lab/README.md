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

## Result table (fill on GPU)

| config | VRAM MiB | TTFT p50 | ITL95 p50 | tok/s | ppl | ppl Δ% |
|---|---|---|---|---|---|---|
| fp16 | | | | | | — |
| fp8 | | | | | | |
| awq | | | | | | |

Success: fp8 within +2% ppl of fp16 with clearly lower VRAM; awq boots the
largest batch (check with `kv_math.py --dtype`... AWQ weights ≈ 3.8 GB, edit
predictor if measured rest-VRAM disagrees by >1 GB).
