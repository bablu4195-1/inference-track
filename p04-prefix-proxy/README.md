# P04 — Prefix-caching proxy + multi-turn A/B

Reused cache is free speed — but only if same-prefix requests reach the same
replica. This project proves both halves: the router, and the TTFT win.

## Parts

| File | Purpose |
|---|---|
| `proxy.py` | stdlib reverse proxy; md5(system-prefix) → replica, round-robin fallback, JSONL route log with proxy-side TTFT |
| `workload.py` | 8 convs × 4 turns, shared ~2k-token system prefix; CSV = shared schema + `turn` |
| `compare.py` | per-turn median TTFT table + cut% bar chart |

## GPU procedure, one boot (~45 min, ~$0.40)

1. Boot with baseline flags (`--enable-prefix-caching`, compose as-is).
   `workload.py --base-url ... --out on.csv` (via proxy or direct).
2. `docker compose down`, add `--no-enable-prefix-caching`, up again.
   `workload.py --base-url ... --out off.csv`.
3. `compare.py on.csv off.csv --plot cut.png` → paste table below.

For multi-replica routing proof (needs 2× GPU — defer to P10 unless rich):
run two servers, point `proxy.py --backends ...`, show `routes.jsonl`
key_hash → backend stability for identical system prompts.

## Result table — first GPU run 2026-09-19 (Qwen2.5-7B, A10G, 8 convs × 4 turns)

| turn | TTFT_on | TTFT_off | cut% |
|---|---|---|---|
| 0 | 802 | 1863 | 57.0% — NOT cold: convs share one system prompt, so convs 1–7 reuse conv 0's blocks (cross-request prefix hit — the proxy's whole job) |
| 1 | 443 | 1800 | 75.4% |
| 2 | 452 | 1611 | 71.9% |
| 3 | 451 | 1620 | 72.2% |

True isolated cold = conv0/turn0 only: 1106 ms (n=1, fresh-server noise —
don't quote it). Lesson for the methodology: shared-prefix workloads make
"turn0" a cross-conversation cache measurement, which is the production shape
anyway. Raw: `runs/20260919-p4/{on,off}.csv`.
