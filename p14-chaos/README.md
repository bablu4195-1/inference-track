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
## Findings — first GPU run 2026-09-20 (kill-replica, A10G)

| Fault | Burn delta | Headroom | MTTR | Masked by |
|---|---|---|---|---|
| kill-replica (`docker restart`) | 100% during window (fast-fail 0.3 s, no hangs) | n/a (single replica, no fallback wired) | **~3.1 min** (kill 13:38:52 → healthy ~13:42) | nothing — and that's the finding: without P13 in front, users eat the full RTO |
| throttle (SM clocks 1710→1005, AWQ) | **+0.0%** — c1 65.2→60.6 tok/s (−7%), c4 52.0→49.8 (−4%) | +1.0pp | n/a (no breach) | memory-bound decode barely notices SM cuts; `-lgc` is the wrong fault for inference — memory clocks (`-lmc`) next time |
| spike | _(pending — c32 absorbed clean in P11 test)_ | | | |

Scaler held correctly through the outage: 34/48 polls metrics-down, zero
scale actions (`hold_metrics_down` — the policy working as designed, no
flapping on blindness). slo.py validated offline; live burn math once the
gateway fronts the fault. Raw: `runs/20260919-p14/`.
