"""Credential loading that never puts a key where a human or a log can see it.

A key reaches this process from a file the user dropped in, or from the
environment. It goes into an ``Authorization`` header and nowhere else. It is
never printed, never written, never returned to a caller, and never placed in a
dictionary that some later ``json.dumps`` might serialise into a report.

The public surface returns *facts about* credentials — which providers are
configured, whether a key authenticates, what balance remains — and never the
credentials themselves. That is the same discipline Leo's own encrypted store
uses, and it exists because the alternative fails silently: a key echoed once
into a log file is compromised whether or not anybody notices.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

#: Files a key might plausibly have been dropped into. Each is gitignored.
_KEY_FILES = ("Openrouter.txt", "openrouter.txt", ".env")

#: OpenRouter keys carry a recognisable prefix, which lets a key be found
#: inside a file that also contains a variable name, quotes or stray notes.
_OPENROUTER_PATTERN = re.compile(r"(sk-or-[A-Za-z0-9_\-]{8,})")


def _extract(text: str) -> str:
    """Pull a key out of whatever shape the file happens to be in."""
    match = _OPENROUTER_PATTERN.search(text)
    if match:
        return match.group(1)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # KEY=value, KEY: value, or a bare key on its own line.
        candidate = re.split(r"[=:]", line, maxsplit=1)[-1].strip()
        candidate = candidate.strip("\"'")
        if candidate:
            return candidate
    return ""


def openrouter_key(root: Path | str = ".") -> str:
    """The key, for immediate use in a header. Never log the return value."""
    from_env = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if from_env:
        return from_env
    root = Path(root)
    for name in _KEY_FILES:
        path = root / name
        if path.is_file():
            key = _extract(path.read_text(encoding="utf-8", errors="replace"))
            if key:
                return key
    return ""


def describe(key: str) -> dict:
    """Facts about a key that are safe to print. Never the key."""
    return {"present": bool(key), "length": len(key),
            "prefix": (key[:7] + "...") if len(key) > 10 else "(too short)"}


def verify_openrouter(key: str) -> dict:
    """Authenticate and report the balance. Returns no secret material."""
    if not key:
        return {"ok": False, "reason": "no key found"}
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = json.load(response).get("data", {})
    except urllib.error.HTTPError as error:
        # The body can echo request details; only the status is reported.
        return {"ok": False, "reason": f"HTTP {error.code}"}
    except Exception as error:  # noqa: BLE001
        return {"ok": False, "reason": type(error).__name__}
    return {
        "ok": True,
        "label": data.get("label"),
        "usage_usd": data.get("usage"),
        "limit_usd": data.get("limit"),
        "remaining_usd": data.get("limit_remaining"),
        "is_free_tier": data.get("is_free_tier"),
    }


def openrouter_models(key: str, contains: str = "") -> list[str]:
    """Model ids currently offered, optionally filtered."""
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    ids = sorted(entry["id"] for entry in data.get("data", []))
    return [i for i in ids if contains.lower() in i.lower()] if contains else ids


def openrouter_pricing(key: str, model_ids: list[str]) -> dict:
    """Per-token prices for named models, so cost can be estimated before spend."""
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    wanted = set(model_ids)
    out = {}
    for entry in data.get("data", []):
        if entry["id"] in wanted:
            pricing = entry.get("pricing", {})
            out[entry["id"]] = {
                "prompt_per_mtok_usd": round(float(pricing.get("prompt", 0)) * 1e6, 4),
                "completion_per_mtok_usd": round(float(pricing.get("completion", 0)) * 1e6, 4),
                "context": entry.get("context_length"),
            }
    return out


__all__ = ["describe", "openrouter_key", "openrouter_models",
           "openrouter_pricing", "verify_openrouter"]
