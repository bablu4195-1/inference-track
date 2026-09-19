# P14 — Chaos suite for inference

Reliability is proven under failure, not in demos. Three faults, each with
auto-recovery, each mapped to the defense that should mask it.

## Fault → expected defense

| Fault (`faults.sh`) | Injects | Should save you | Proves |
|---|---|---|---|
| `kill-replica` | `docker restart` (~60–90 s dark) | P13 fallback chain | RTO < 2 min, zero user errors |
| `throttle-gpu` | clocks halved, 120 s, auto-restore | P8-tuned budgets + P11 scale | graceful degradation, not cliff |
| `spike` | 64-conc blast via P2 | P11 scale-up + P13 429s | shed load, never 503 |

`--dry-run` default prints; `--exec` injects. Throttle always auto-restores
(even the dry-run tells you so) — a forgotten chaos session must never burn
the $128 envelope or cook the GPU.

## Measuring (`slo.py`)

Error budget from results CSVs: a request burns if `ok=False`, TTFT > 2 s, or
ITL-p95 > 500 ms. Baseline vs fault window → **burn delta**, budget headroom
vs 99% target, MTTR proxy, per-bucket recovery curve.

## GPU procedure, one boot (~45 min, ~$0.40)

Gateway (P13) up fronting vLLM; scaler (P11) watching. Baseline P2 quick →
each fault + P2 quick during fault → `slo.py` per fault. Bonus: tonight's
accidental spot reclaim is already filed as chaos evidence — see boot log.
## Findings (fill on GPU)

| Fault | Burn delta | Headroom | MTTR | Masked by |
|---|---|---|---|---|
| kill-replica | | | | |
| throttle | | | | |
| spike | | | | |
