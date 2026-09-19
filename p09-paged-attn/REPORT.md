# P09 — PagedAttention under memory pressure: deep-dive report

## 1. How vLLM paging works (the 2-minute primer)

The KV cache is split into fixed-size **blocks** (`--block-size` tokens each,
default 16). A request's blocks need not be contiguous in GPU memory — a
**block table** maps logical token positions to physical blocks, exactly like
OS virtual memory pages. Consequences:

- **No external fragmentation by design.** A 6k-token request never needs a
  contiguous 6k-token slab; any free blocks do. Allocation fails only when the
  pool is truly exhausted, never because free memory is "in the wrong shape".
- **Internal fragmentation remains.** The last block of each sequence is
  partially empty: average waste ≈ `block_size / 2` tokens per sequence.
  Waste fraction ≈ `block_size / (2 × seq_len)`:

  | seq_len | bs=8 | bs=16 | bs=32 |
  |---|---|---|---|
  | 128 | 3.1% | 6.3% | 12.5% |
  | 2048 | 0.2% | 0.4% | 0.8% |
  | 6000 | 0.07% | 0.13% | 0.27% |

  Small block sizes waste less per sequence but inflate the block table and
  kernel launch overhead; large blocks waste more on short requests. The
  matrix below measures which effect dominates on real hardware.

## 2. Methodology

- Model Qwen2.5-7B FP16, A10G 24 GB, `--gpu-memory-utilization 0.9`.
- Pressure workload: 6 waves × conc 32 × 6k-token prompts (~10.7 GB KV demand
  vs ~6.4 GB headroom — over capacity by design).
- Distress sequence watched in `/metrics`: `kv_cache_usage_perc` → 100%,
  then `num_requests_waiting` > 0 (queueing), then `num_requests_swapped` > 0
  (preemption: lowest-priority sequence evicted, its KV dropped, recomputed
  on reschedule — the latency tax of paging).
- Configs differ **only** in `--block-size` (8/16/32); all else pinned
  (see `run-matrix.sh`). Predictor says all three OOM — the question is
  *how gracefully* and at what throughput cost.

## 3. Results (fill on GPU — `analyze.py --report` output goes here)

| label | wait@wave | swap@wave | err% | tok/s | peak_kv |
|---|---|---|---|---|---|
| bs8 | | | | | |
| bs16 | | | | | |
| bs32 | | | | | |

## 4. Findings (fill on GPU)

- **Eviction onset:** which block size queues/swaps first, and by how many waves?
- **Graceful vs cliff:** does throughput degrade smoothly (queueing absorbs)
  or fall off a cliff (mass eviction + recompute storms)? Quote wave tok/s.
- **Fragmentation verdict:** does bs=8's lower waste buy measurable capacity
  (later first_swap), or does table/kernel overhead dominate (lower tok/s)?
- **Scheduler lesson:** the flag that mattered most under pressure was ___,
  because ___.

## 5. Repro

`./p09-paged-attn/run-matrix.sh` on any 24 GB GPU (~45 min spot, ~$0.40).
Raw timelines: `results/<label>-*/summary.json`.
