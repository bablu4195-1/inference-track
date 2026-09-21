# The Inference Track, Explained for Beginners

*You don't need to know what a GPU kernel is. By the end of this page, you will.*

## The 30-second background

A large language model (LLM) is a giant math function. When you chat with one,
your computer runs that function over and over — once per word it writes.
**Inference** is the job of running that function fast, cheap, and reliably
for real users. This repo is a 15-project journey through that job, built and
measured on real rented GPUs for under $10 so far.

Two words you'll see everywhere:

- **Prefill** — reading your prompt all at once (like speed-reading a page).
  Heavy math, uses the GPU's calculators hard.
- **Decode** — writing the answer one word at a time (like typing). Each step
  needs to re-read everything written so far, which lives in a scratchpad
  called the **KV cache**. Decode is usually waiting on memory, not math.

Keep that split in your head — half of these projects are about it.

**Glossary in one breath:** *VRAM* = the GPU's own fast memory (24 GB on our
card). *TTFT* = time to first token (how long you stare at "..."). *ITL* =
inter-token latency (the rhythm words appear in). *Throughput* = total words
per second. *OOM* = out of memory, the crash everyone fears.

---

## P1 — Self-Hosted Inference Server
*Running someone else's recipe in your own kitchen.*

Instead of paying OpenAI per word, we rented a GPU and served an open model
(Qwen2.5-7B) ourselves with **vLLM**, the standard open-source serving engine.
vLLM's superpower is **continuous batching**: it doesn't wait for one user's
answer to finish before starting the next — it weaves many users' words into
every GPU heartbeat, like a chef working five pans at once.

- **Built:** one-command server (`p01-serve/compose.yml`), health checks, a
  2-user smoke test.
- **Found:** first real answer in ~480 ms. It works. Everything else rests on this.

## P2 — TTFT/ITL Benchmark Suite
*A speedometer you trust, because you built it.*

Vendors publish latency numbers measured under perfect conditions. We wrote
our own harness (`p02-bench/bench.py`) that fires hundreds of requests at
rising concurrency and records TTFT, ITL, and throughput per request — the
same tool every later project reuses, so all numbers are comparable.

- **Built:** async load generator, 9-scenario matrix (3 prompt sizes × 3
  concurrency levels), CSV + load-curve plots.
- **Found:** the classic "knee" — fine up to ~8 simultaneous users, then TTFT
  climbs 551 ms → 2 s. Also caught a real bug: 32 simultaneous connections
  tripped the server during startup compilation; retry + request staggering
  took errors 93 → 0.

## P3 — KV Cache Calculator and Monitor
*Doing the memory math before the crash does it for you.*

The KV scratchpad grows with every word of every user. Our predictor
(`libs/kv_math.py`) computes exactly how many gigabytes any workload needs
from the model's architecture — no guessing. A live monitor watches the real
server's gauges and shouts before the crash.

- **Built:** GQA-aware VRAM formula, CLI (`--table` prints capacity per
  prompt length), live `/metrics` monitor with a 90% alarm, and a pressure
  procedure that ramps users until distress.
- **Found:** prediction within **10% of reality** (predicted 10 simultaneous
  long conversations; observed 11). The detector itself taught us two lessons:
  measure *during* the storm, not after (queues drain instantly), and use
  *different* prompts per user (identical prompts share memory and measure nothing).

## P4 — Prefix Caching Proxy
*Don't re-read the instructions every time.*

Every chat with the same app starts with the same system prompt ("You are a
helpful assistant..."). Recomputing its scratchpad for every user is pure
waste — unless requests land on *different* servers, which is what naive
load-balancing does. Our proxy (`p04-prefix-proxy/proxy.py`) fingerprints the
shared prefix and routes same-prefix requests to the same machine, so the
cached scratchpad gets reused.

- **Built:** zero-dependency routing proxy (consistent hashing + round-robin
  fallback), a multi-turn test workload, an A/B comparison tool.
- **Found:** **72–75% faster first words** on later conversation turns.
  Subtlety discovered: "turn 0" isn't really cold when conversations share one
  system prompt — cross-user reuse *is* the feature.

## P5 — Quantization Comparison Lab
*Same brain, smaller suitcase.*

Model weights are normally stored as 16-bit numbers. **Quantization** squeezes
them into 8 or 4 bits — smaller, faster to load, with (hopefully) no visible
quality loss. We served the same model three ways and measured quality
(perplexity — how "surprised" the model is by real text), speed, and memory.

- **Built:** 3-config serve matrix, a perplexity probe needing zero extra
  software (it reads the server's own word-probabilities), a tradeoff table.
- **Found:** **AWQ wins clearly** — 2% quality cost, ~2× speed, half the
  memory wait. Two honest failures kept: full FP8 mode *crashes* on our GPU
  generation (documented, with error messages), and uncalibrated FP8 memory
  mode wrecked quality 5×. Negative results are results.

## P6 — Speculative Decoding Pipeline
*Let the intern draft, the expert approve.*

A tiny 0.5B model guesses the next 5 words (fast, often right); the big 7B
model checks all 5 in one pass and keeps the good ones. Output is
*provably identical* to the big model alone — a genuine free lunch, limited
only by how often the intern is right (**acceptance rate**).

- **Built:** 4-config matrix (off + 3/5/8 draft words), acceptance tracking
  scraped from server metrics, speedup comparison.
- **Found:** **2.26× speedup at 96% acceptance** with 5 draft words; 8 words
  gets worse (more guessing, less checking). Three live lessons: new server
  versions renamed the flags, the two models' dictionaries differ slightly
  (needs a compatibility flag), and speed measurement itself had to be fixed
  (the server bundles words per update — count words, not updates).

## P7 — Triton Kernel from Scratch
*Opening the hood of the GPU.*

Every model spends much of its time in tiny math operations like RMSNorm
(rescaling each row of numbers). The default implementation makes 4 separate
trips to memory; our hand-written GPU **kernel** fuses them into one —
same math, ~3× less memory traffic. Written in Triton (Python-like GPU
language), checked against the reference to 4 decimal places.

- **Built:** fused RMSNorm kernel, CPU test suite (5 tests, all passing on a
  Mac with no GPU), benchmark harness.
- **Found:** **3.6× faster**, correctness 10× better than required. Caveat
  kept: the GB/s headline flatters both sides equally (test data fits in
  cache) — the *ratio* is the honest number.

## P8 — Chunked Prefill Scheduler Experiment
*Don't let one long reader starve the typists.*

One user pasting an 8,000-word document forces a giant math chunk that can
stall everyone else's word-by-word typing. **Chunked prefill** slices the big
read into pieces interleaved with typing. We varied the slice budget with the
feature on and off, expecting a dramatic rescue.

- **Built:** mixed-load experiment (decoders + one giant hog), 5-config
  restart matrix, tradeoff plots.
- **Found:** an **honest null** — no starvation either way, because the modern
  scheduler already protects typists. The real knob turned out to be the
  *budget size itself* (46 → 97 ms per word from smallest to largest). Kept,
  documented, retested at 3× pressure to be sure.

## P9 — PagedAttention Deep-Dive Report
*Virtual memory, but for AI.*

The KV scratchpad is chopped into fixed **blocks** (like OS memory pages), so
requests never need one giant unbroken slab — this kills an entire class of
memory crashes *by design*. We hammered the server past 100% memory with
three block sizes to watch fragmentation and evictions (kicking users out)
happen.

- **Built:** over-capacity pressure driver, eviction-timeline analysis,
  block-size matrix, full written report (`REPORT.md`).
- **Found:** another honest null with a great moral — block sizes 16 vs 32
  were indistinguishable, tiny blocks are *rejected* by modern software, and
  **the server never evicted anyone**: at 100% memory it queues gracefully
  instead of crashing. The failure mode is a polite line, not a cliff.

## P10 — Disaggregated Prefill/Decode Cluster
*Two kitchens: one for reading, one for typing.*

Frontier labs split the two phases onto separate GPU pools so readers and
typists can never starve each other — connected by special software (NIXL)
that beams the scratchpad from reader to typist mid-request. We built it:
two workers, a routing proxy, and measured it against the single-kitchen
setup on one GPU and two.

- **Built:** dual-worker configs, a from-scratch routing proxy
  (`mini_proxy.py`), a 9-scenario comparison. Debugging the handshake took
  the longest: deprecated settings, invisible network addresses, and a proxy
  bug that sent workers dialing themselves.
- **Found:** real cross-machine transfers, zero errors across 246 requests —
  but the single setup won 9–0 on speed here, because our test never
  saturated it (splitting only pays when the single kitchen is overwhelmed).
  Correctly scoped to a future high-pressure rerun.

## P11 — Queue-Based GPU Autoscaler
*A thermostat for GPUs.*

Idle GPUs burn money; overloaded ones burn users. Our controller watches one
number — how many requests are *waiting* — and adds a server after sustained
pressure, removes it after sustained calm, with cooldowns so it never
flaps. Replica #1 never sleeps (a warm spare starts in 30 seconds; a cold
one takes 8 minutes).

- **Built:** control loop with unit tests (14 checks, and the tests caught a
  real double-counting bug before it could flap a fleet), scale-event log.
- **Found:** a **full live round trip** — scaled 1→2 fifty-seven seconds into
  real queue pressure, back to 1 two minutes after drain. Plus proof of the
  harder virtue: it correctly did *nothing* during a traffic spike the server
  absorbed and during a total outage (never scale on blindness).

## P12 — Cost-per-Token Dashboard
*Inference is won on unit economics.*

We turned every test result in the repo plus the real spend ledger into
dollars-per-million-words and hardware-efficiency (MFU) per workload, with
costs split by actual GPU-time consumed, not naive word counts.

- **Built:** accounting script (reads any results file + the spend log),
  4-panel dashboard image, per-tenant tracking hooks.
- **Found:** our scale runs cost **$0.59 per million words**; long documents
  are cheapest per word (density beats compute growth). Total track spend:
  **under $10 of a $128 budget**, every cent logged.

## P13 — AI Gateway with Fallbacks and Rate Limits
*The receptionist who never panics.*

One door for all users, standing in front of every server: if the main model
is slow or dead, requests automatically fail over to a backup (users never
see the outage); each customer gets a fair-use token bucket (429 "slow down"
instead of a crash); every decision is logged for the accountants in P12.

- **Built:** stdlib-only gateway (failover chain, first-word deadlines,
  retries, per-customer rate limits), 16 automated tests.
- **Found:** testing caught a real bug — a missed deadline was mislabeled and
  wasted a retry; fixed to fail over immediately (halved recovery time).

## P14 — Chaos Suite for Inference
*Breaking things on purpose, on schedule.*

Three disasters with automatic recovery: killing a server mid-traffic,
halving the GPU's clock speed, and a 64-user stampede — each mapped to the
defense that should absorb it, measured as budget-burn against our
reliability targets.

- **Built:** three fault injectors (all self-healing — a forgotten chaos test
  can't burn money or cook hardware), an SLO-burn reporter.
- **Found:** killing a server costs **3.1 minutes** of fast-failing errors
  with no backup wired (the case *for* P13, in one number); halving GPU speed
  cost only 4–7% speed (typing waits on memory, not math — wrong fault for
  this workload, noted for next time). Bonus: the cloud provider killed *our*
  servers twice for real — filed as field data.

## P15 — Public Benchmark Teardown
*Show your work, with the ugly parts.*

The finale: three full serving setups raced on identical work, published with
every config, script, and raw CSV needed to reproduce them — including a
flagged outlier we didn't explain away and an appendix of everything that
broke along the way.

- **Built:** 3-way comparison (stock vs AWQ vs speculative), one-command
  repro script, this report.
- **Found:** AWQ dominates almost everywhere (+100–260% throughput *and*
  faster first words); speculative decoding roughly doubles speed with one
  caveat (can lose first-word time at low load). The document argues a good
  benchmark teardown is the strongest hiring signal in infrastructure work —
  this repo is that argument, with receipts.

---

*Built over ~9 GPU sessions, ~$10 of a $128 budget, 25+ commits. Every number
above links to a CSV in `runs/`. Where the hypothesis failed, it says so —
that's the point.*
