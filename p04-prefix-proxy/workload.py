#!/usr/bin/env python3
"""P4 multi-turn workload: shared system prompt + varying user turns.

Sends C conversations x T turns. Each turn's prompt = SYSTEM (shared ~2k
tokens, the cacheable prefix) + history + new user turn — exactly the shape
that benefits from prefix caching on turn >= 2.

Output CSV = shared schema + turn column. Run once per server config, then
compare.py --a cache-on.csv --b cache-off.csv.

Usage:
  python3 p04-prefix-proxy/workload.py --base-url http://<GPU>:8000 --out on.csv
  # restart server with --no-enable-prefix-caching, then:
  python3 p04-prefix-proxy/workload.py --base-url http://<GPU>:8000 --out off.csv
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "libs"))
sys.path.insert(0, str(ROOT / "p02-bench"))

import bench  # noqa: E402
from metrics import CSV_HEADER  # noqa: E402
from prompts import FILLER  # noqa: E402
from vllm_client import health  # noqa: E402

SYSTEM_TOKENS = 2000
USER_TURNS = [
    "Summarize the key tradeoffs in one paragraph.",
    "Now give a concrete numeric example.",
    "What breaks first under 10x load?",
    "List three follow-up experiments.",
]


def system_prompt() -> str:
    need = SYSTEM_TOKENS * 4
    reps = (need // len(FILLER)) + 1
    return "[shared system prompt]\n" + (FILLER * reps)[:need]


async def conversation(session: aiohttp.ClientSession, base_url: str, model: str,
                       gen: int, timeout_s: float, system: str, cid: int):
    """One sequential multi-turn conversation -> list of row dicts."""
    rows = []
    history = ""
    for turn, user in enumerate(USER_TURNS):
        prompt = f"{system}\n{history}User: {user} (conv {cid})\nAssistant:"
        stats, ok, err = await bench.one_request(
            session, base_url, model, prompt, gen, timeout_s)
        rows.append({
            "run_id": "", "config": f"turn{turn}", "concurrency": 1,
            "prompt_len": len(prompt) // 4, "gen_len": stats.gen_tokens,
            "ttft_ms": round(stats.ttft_ms, 1),
            "itl_p50_ms": round(stats.itl_p50_ms, 2),
            "itl_p95_ms": round(stats.itl_p95_ms, 2),
            "e2e_ms": round(stats.e2e_ms, 1),
            "toks_per_s": round(stats.toks_per_s, 2),
            "ok": ok, "err": err, "turn": turn, "conv": cid,
        })
        history += f"User: {user}\nAssistant: <{stats.gen_tokens} tokens>\n"
    return rows


async def amain(a: argparse.Namespace) -> int:
    if not await health(a.base_url):
        print(f"FATAL: {a.base_url}/health unreachable", file=sys.stderr)
        return 1
    system = system_prompt()
    print(f"system prefix ~{len(system)//4} tokens, "
          f"{a.convs} convs x {len(USER_TURNS)} turns")
    all_rows = []
    async with aiohttp.ClientSession() as session:
        # Conversations run concurrently (realistic gateway load); turns within
        # a conversation stay sequential (causal history).
        for batch in range(0, a.convs, a.conc_convs):
            group = [conversation(session, a.base_url, a.model, a.gen,
                                  a.timeout_s, system, cid)
                     for cid in range(batch, min(batch + a.conc_convs, a.convs))]
            for rows in await asyncio.gather(*group):
                all_rows.extend(rows)
            time.sleep(a.cooldown_s)
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER + ["turn", "conv"])
        w.writeheader()
        w.writerows(all_rows)
    n = len(all_rows)
    print(f"WROTE {a.out} ({n} rows, {sum(1 for r in all_rows if not r['ok'])} errors)")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="P4 multi-turn workload")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--convs", type=int, default=8)
    ap.add_argument("--conc-convs", type=int, default=4,
                    help="concurrent conversations")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--cooldown-s", type=float, default=2.0)
    ap.add_argument("--timeout-s", type=float, default=600.0)
    a = ap.parse_args()
    sys.exit(asyncio.run(amain(a)))


if __name__ == "__main__":
    main()
