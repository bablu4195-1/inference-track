# P08 — Chunked-prefill scheduler experiment

Long-context prefills are compute-bound; decodes are memory-bound. Without
scheduling control, one 8k hog stalls every decode stream. vLLM's answer:
chunk the hog's prefill into the `max-num-batched-tokens` budget alongside
decodes (on by default in V1, tunable via budget size).

## Parts

| File | Purpose |
|---|---|
| `experiment.py` | baseline-decode phase + mixed hog+decode phase → `starvation_x` |
| `run-matrix.sh` | 5 server configs, sequential restarts, one boot (~1h, ~$0.55) |
| `compare.py` | tradeoff table + starvation/TTFT plots |

## Matrix

| label | chunked | budget | Expectation |
|---|---|---|---|
| on-2048 | on | 2048 | kindest decodes, slowest hog |
| on-4096 | on | 4096 | |
| on-8192 | on | 8192 | balanced |
| on-16384 | on | 16384 | hog fast, decodes pressured |
| off-16384 | **off** | 16384 | decode starvation (control) |

Tuning rule of thumb (vLLM docs): smaller budget → better ITL, worse TTFT;
throughput wants budget > 8192 on big GPUs. The experiment finds the crossing
for *this* model/GPU/workload instead of quoting docs.

## Result table (fill on GPU)

| label | ITL_base | ITL_mixed | starvation_x | hog_TTFT |
|---|---|---|---|---|
| on-2048 | | | | |
| on-4096 | | | | |
| on-8192 | | | | |
| on-16384 | | | | |
| off-16384 | | | >>1 expected | |

Success: `off-16384` starvation clearly > all `on-*`, and a monotonic
budget trend across `on-*`. If `off` looks identical to `on-16384`, the hog
was too small — raise `--hog-tokens` (needs `--max-model-len` headroom).
