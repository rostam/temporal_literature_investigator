"""LLM backend abstraction.

Two interchangeable backends, selected by TLI_LLM_BACKEND:
  - "cli"  (default): shells out to the authenticated `claude -p` CLI — no API
    key needed, uses your Claude Code login. Good for getting started.
  - "sdk": the anthropic SDK directly (needs ANTHROPIC_API_KEY). Lower latency,
    supports token streaming and prompt caching.

Both expose complete()/stream(system, user, *, model, max_tokens) -> str.
Note: the CLI backend ignores max_tokens (the CLI manages its own) and streams
only at the response level, not token-by-token.
"""
from __future__ import annotations

import json
import subprocess
from functools import lru_cache

from . import config


class LLMError(RuntimeError):
    pass


class CLIBackend:
    name = "cli"

    def complete(self, system: str, user: str, *, model=None, max_tokens=None) -> str:
        cmd = ["claude", "-p", "--output-format", "json"]
        if system:
            cmd += ["--system-prompt", system]
        if model:
            cmd += ["--model", model]
        try:
            r = subprocess.run(cmd, input=user, capture_output=True, text=True)
        except FileNotFoundError:
            raise LLMError(
                "`claude` CLI not found on PATH. Install Claude Code, or set "
                "TLI_LLM_BACKEND=sdk with an ANTHROPIC_API_KEY."
            )
        if r.returncode != 0:
            raise LLMError(f"claude -p failed (exit {r.returncode}): "
                           f"{r.stderr.strip() or r.stdout.strip()}")
        try:
            d = json.loads(r.stdout)
        except json.JSONDecodeError:
            raise LLMError(f"Unexpected output from claude -p: {r.stdout[:300]}")
        if d.get("is_error"):
            raise LLMError(f"claude -p returned an error: {d.get('result')}")
        return d.get("result", "")

    def stream(self, system: str, user: str, *, model=None, max_tokens=None) -> str:
        # CLI backend has no token streaming here; print the full response.
        text = self.complete(system, user, model=model, max_tokens=max_tokens)
        print(text)
        return text


class SDKBackend:
    name = "sdk"

    def __init__(self):
        import anthropic  # lazy: only needed for this backend

        self._client = anthropic.Anthropic()

    def _kwargs(self, system, user, model, max_tokens):
        return dict(
            model=model or config.SYNTH_MODEL,
            max_tokens=max_tokens or 2000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": system,
                     "cache_control": {"type": "ephemeral"}}] if system else None,
            messages=[{"role": "user", "content": user}],
        )

    def complete(self, system, user, *, model=None, max_tokens=None) -> str:
        msg = self._client.messages.create(
            **{k: v for k, v in self._kwargs(system, user, model, max_tokens).items()
               if v is not None})
        return "".join(b.text for b in msg.content if b.type == "text")

    def stream(self, system, user, *, model=None, max_tokens=None) -> str:
        out = []
        kwargs = {k: v for k, v in self._kwargs(system, user, model, max_tokens).items()
                  if v is not None}
        with self._client.messages.stream(**kwargs) as s:
            for text in s.text_stream:
                print(text, end="", flush=True)
                out.append(text)
        print()
        return "".join(out)


@lru_cache(maxsize=1)
def get_llm():
    if config.LLM_BACKEND == "sdk":
        return SDKBackend()
    return CLIBackend()
