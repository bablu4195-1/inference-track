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

## Validation table (fill from `validation-<utc>.json`)

| seq_len | Predicted max_batch | Observed stable | err% | Verdict |
|---|---|---|---|---|
| 2048 | | | | |
| 4096 | | | | |
| 8192 | | | | |

If err% > 15%: first suspect `--gpu-memory-utilization` and block-size
overhead (`BLOCK_OVERHEAD` in `kv_math.py`), not the formula — then recalibrate.
