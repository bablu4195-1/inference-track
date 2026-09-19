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

## Result table (fill on GPU)

| turn | TTFT_on | TTFT_off | cut% |
|---|---|---|---|
| 0 (cold) | | | ~0% expected |
| 1 | | | |
| 2 | | | |
| 3 | | | |

Success bar: turn ≥ 1 cut ≥ 40% on the ~2k shared prefix. If turn0 shows a
cut, something's wrong (cold prefix can't hit) — check server flags.
