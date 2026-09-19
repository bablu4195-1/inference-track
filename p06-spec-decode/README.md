# P06 — Speculative decoding pipeline

Small draft (Qwen2.5-0.5B, ~1 GB) proposes k tokens; the 7B target verifies
them in one forward pass. Accepted prefix is kept, rejected tail is resampled —
output distribution is **exact**, so any tok/s gain is a free lunch. The only
question is how often the draft is right (acceptance rate).

## Matrix (one boot, ~1h, ~$0.55)

| config | flags | Expectation |
|---|---|---|
| baseline | — | reference tok/s |
| spec-3 | `--speculative-model …0.5B --num-speculative-tokens 3` | modest gain, high acceptance |
| spec-5 | … tokens 5 | best tradeoff (typical) |
| spec-8 | … tokens 8 | more draft work, acceptance falls |

Quality matched by construction: greedy decoding (temp 0), same P2 prompts —
spec decoding provably preserves the target distribution, so no quality eval
is needed. If tok/s doesn't move but acceptance is high, suspect draft/target
tokenizer mismatch or tiny batch (spec wins grow with batch).

## Result table (fill on GPU)

| config | tok/s | speedup | ITL95 | acceptance |
|---|---|---|---|---|
| baseline | | 1.00x | | — |
| spec-3 | | | | |
| spec-5 | | | | ≥60% target |
| spec-8 | | | | |

Success bar: **≥1.3× decode speedup with acceptance ≥60%** on at least one
draft size. `acceptance.py` scrapes `vllm:spec_decode*` by prefix (survives
metric renames); the bench speedup is the proof, acceptance is the explanation.
