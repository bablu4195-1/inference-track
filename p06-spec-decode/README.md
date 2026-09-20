# P06 — Speculative decoding pipeline

Small draft (Qwen2.5-0.5B, ~1 GB) proposes k tokens; the 7B target verifies
them in one forward pass. Accepted prefix is kept, rejected tail is resampled —
output distribution is **exact**, so any tok/s gain is a free lunch. The only
question is how often the draft is right (acceptance rate).

## Matrix (one boot, ~1h, ~$0.55)

| config | flags (vLLM 0.29 form) | Expectation |
|---|---|---|
| baseline | — | reference tok/s |
| spec-3 | `--speculative-config '{"method":"draft_model","model":"…0.5B","num_speculative_tokens":3}'` | modest gain, high acceptance |
| spec-5 | … tokens 5 | best tradeoff (typical) |
| spec-8 | … tokens 8 | more draft work, acceptance falls |

(Legacy `--speculative-model/--num-speculative-tokens` flags are gone in
0.29 — caught live on the GPU, script updated.)

Quality matched by construction: greedy decoding (temp 0), same P2 prompts —
spec decoding provably preserves the target distribution, so no quality eval
is needed. If tok/s doesn't move but acceptance is high, suspect draft/target
tokenizer mismatch or tiny batch (spec wins grow with batch).

## Result table — first GPU run 2026-09-20 (Qwen2.5-7B + 0.5B draft, A10G, P2 quick)

| config | tok/s | speedup | ITL95 | acceptance |
|---|---|---|---|---|
| baseline | 45 | 1.00x | 147.4 | — |
| spec-3 | 64 | 1.42x | 136.6 | 97% |
| spec-5 | 101 | 2.26x | 100.7 | 96% |
| spec-8 | 71 | 1.59x | 125.9 | 94% |

Three live lessons: (1) legacy spec flags are gone in 0.29 —
`--speculative-config` JSON only; (2) 0.5B draft vocab (151936) ≠ 7B target
(152064) — needs `use_heterogeneous_vocab`; (3) spec batches tokens per SSE
event — chunk-counting understated throughput ~2× until `stream_options`
usage counts (fixed in bench + vllm_client). Raw: `runs/20260919-p6/`.
