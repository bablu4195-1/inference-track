#!/usr/bin/env python3
"""P5 quality probe: perplexity from vLLM prompt logprobs — zero extra deps.

Sends chunks of sample.txt to /v1/completions with prompt_logprobs and sums
log P(token) -> ppl = exp(-mean logprob). Same text for every config, so any
sample bias cancels in the comparison (documented in README, not hidden).

The first token of each chunk lacks full context (causal scoring boundary);
we drop the first --skip tokens per chunk to avoid polluting the mean.

Usage: python3 p05-quant-lab/quality.py --base-url URL --label fp16
Writes results/<label>-<utc>/ppl.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import urllib.request
from pathlib import Path


def score_chunk(base_url: str, model: str, text: str, timeout_s: float) -> list[float]:
    body = json.dumps({"model": model, "prompt": text, "max_tokens": 1,
                       "temperature": 0.0, "prompt_logprobs": 1}).encode()
    req = urllib.request.Request(f"{base_url}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read().decode())
    plps = data["choices"][0].get("prompt_logprobs") or []
    out = []
    for tok in plps:
        if not tok:
            continue  # boundary / skipped token
        # {logprob: float} top-1 entry (or {token_id: {...}} shape — take first)
        v = next(iter(tok.values()))
        lp = v.get("logprob", v) if isinstance(v, dict) else v
        out.append(float(lp))
    return out


def chunk_words(words: list[str], n: int):
    return [" ".join(words[i:i + n]) for i in range(0, len(words), n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="P5 perplexity probe")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--label", required=True)
    ap.add_argument("--text", default=str(
        Path(__file__).resolve().parent / "sample.txt"))
    ap.add_argument("--words-per-chunk", type=int, default=120)
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    ap.add_argument("--out-dir", default=str(
        Path(__file__).resolve().parent / "results"))
    a = ap.parse_args()

    words = Path(a.text).read_text().split()
    chunks = chunk_words(words, a.words_per_chunk)
    print(f"{len(words)} words in {len(chunks)} chunks vs {a.base_url}")
    lps: list[float] = []
    for i, ch in enumerate(chunks):
        got = score_chunk(a.base_url, a.model, ch, a.timeout_s)[a.skip:]
        lps.extend(got)
        print(f"  chunk{i}: {len(got)} scored tokens")
    ppl = math.exp(-sum(lps) / len(lps))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(a.out_dir) / f"{a.label}-{stamp}"
    d.mkdir(parents=True, exist_ok=False)
    (d / "ppl.json").write_text(json.dumps({
        "label": a.label, "run_utc": stamp, "model": a.model,
        "tokens_scored": len(lps), "mean_logprob": sum(lps) / len(lps),
        "perplexity": round(ppl, 3)}, indent=2))
    print(f"perplexity={ppl:.3f} on {len(lps)} tokens -> {d}/ppl.json")


if __name__ == "__main__":
    main()
