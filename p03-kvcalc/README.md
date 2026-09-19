# P03 — KV-cache calculator + live monitor

Most production OOMs are KV math errors, not model size. This project makes
the math explicit, then checks it against a live server.

## Parts

| File | Purpose |
|---|---|
| `../libs/kv_math.py` | predictor: KV bytes/token, max batch per seq-len (GQA-aware) |
| `monitor.py` | polls `/metrics`, logs KV/queue/swap gauges, ALERTs at 90% |
| `pressure.py` | ramps concurrency past the prediction until distress; PASS if within ±15% |

Qwen2.5-7B reference: 56 KiB/token; seq-8192 tops out at batch ~10 in FP16
on a 24 GB A10G at `--gpu-memory-utilization 0.9`.

## Distress definition (what counts as "prediction failed")

Stop the ramp when **either** fires: any `ok=False` request in the wave, or
`vllm:num_requests_waiting >= 4` (scheduler pressure precedes eviction).
`monitor.py` in a second terminal timestamps the same event independently.

## GPU procedure (~30–45 min, ~$0.40)

Terminal 1: `python3 p03-kvcalc/monitor.py --base-url http://<GPU>:8000`
Terminal 2: `python3 p03-kvcalc/pressure.py --base-url http://<GPU>:8000`

## Validation table

| seq_len | Predicted max_batch | Observed | err% | Verdict |
|---|---|---|---|---|
| 8192, distinct prompts (2026-09-19, cache ON) | 10 | 11 (queueing from conc 10, distress at 12) | +10.0% | **PASS** |
| 8192, identical prompts (2026-09-19) | 10 | ≥14, KV only 12% | — | invalid run: shared prefix, kept as lesson |

Two lessons kept: (1) post-wave metric scrapes are blind — `sample_during()`
scrapes mid-wave; (2) capacity ramps need distinct prompts — identical prompts
share one prefix block (that run measured prefix-cache efficiency: 14×8k in
12% KV). KV slope re-validates per-token math: ~10.4% pool/conc ≈ 65–70 KB/tok
vs 61.6 predicted. Raw: `runs/20260919-p3/`.

If err% > 15%: first suspect `--gpu-memory-utilization` and block-size
overhead (`BLOCK_OVERHEAD` in `kv_math.py`), not the formula — then recalibrate.
