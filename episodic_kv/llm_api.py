"""DeepSeek API client (OpenAI-compatible) for end-to-end LLM evaluation.

Set ``DEEPSEEK_API_KEY`` in the environment or in a local ``.env`` file
(never commit real keys). Default model: ``deepseek-v4-flash``.
"""
from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"


def _load_dotenv():
    """Load ``.env`` from project root if present."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def get_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Export it or add to .env (see .env.example)."
        )
    return key


def chat(
    messages: list[dict[str, str]],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
    thinking: str = "disabled",
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 120.0,
    retries: int = 3,
) -> dict[str, Any]:
    """Call DeepSeek chat/completions. Returns parsed JSON response."""
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": {"type": thinking},
        "stream": False,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {get_api_key()}",
        },
        method="POST",
    )
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"HTTP {e.code}: {err_body[:500]}")
            if e.code in (429, 500, 502, 503) and attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            raise last_err from e
        except Exception as e:
            last_err = e
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise last_err  # type: ignore[misc]


def complete(
    prompt: str,
    system: str = "You are a precise QA assistant. Answer briefly.",
    **kwargs,
) -> str:
    """Single-turn completion; returns assistant text."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    resp = chat(messages, **kwargs)
    return resp["choices"][0]["message"]["content"].strip()


def ping() -> dict[str, Any]:
    """Smoke test the API."""
    text = complete("Reply with exactly: OK", max_tokens=16)
    return {"model": DEFAULT_MODEL, "reply": text}
