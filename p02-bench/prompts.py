"""Deterministic synthetic prompts for load testing.

Content-independent latency for dense models: what matters is token COUNT,
not text. We approximate 1 token ~= 4 chars (Llama/Qwen BPE on English prose)
and stamp the exact target so runs are reproducible. P15 methodology notes
the approximation; relative comparisons (the point of P2) are unaffected.
"""

from __future__ import annotations

FILLER = (
    "The quick brown fox jumps over the lazy dog. Paged attention splits the "
    "key value cache into fixed blocks so concurrent requests share GPU memory "
    "without fragmentation. Continuous batching merges prefill and decode work "
    "into every forward pass. "
)

CHARS_PER_TOKEN = 4


def make_prompt(target_tokens: int, seed: int = 0) -> str:
    header = f"[bench seed={seed} target_tokens={target_tokens}]\n"
    need = target_tokens * CHARS_PER_TOKEN - len(header)
    reps = (need // len(FILLER)) + 1
    body = (FILLER * reps)[:need]
    return header + body


def prompt_len_tokens(prompt: str) -> int:
    return len(prompt) // CHARS_PER_TOKEN
