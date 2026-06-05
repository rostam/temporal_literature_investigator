"""Central configuration. Reads .env, resolves paths, picks models/backends."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Repo root = two levels up from this file (src/tli/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# --- Source corpus (the Jekyll blog) ---
BOOKS_REPO = _resolve(os.getenv("TLI_BOOKS_REPO", "../travel-in-books.github.io"))
POSTS_DIR = BOOKS_REPO / "_posts"

# --- Generated artifacts ---
DATA_DIR = REPO_ROOT / "data"
BOOKS_JSONL = DATA_DIR / "books.jsonl"
HISTORY_DIR = DATA_DIR / "history"
PROFILES_DIR = DATA_DIR / "profiles"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "tli"

# --- LLM backend: "cli" (authenticated `claude -p`, no key) or "sdk" (API key) ---
LLM_BACKEND = os.getenv("TLI_LLM_BACKEND", "cli").lower()

# --- Models (see the claude-api skill for current IDs) ---
SYNTH_MODEL = os.getenv("TLI_SYNTH_MODEL", "claude-opus-4-8")     # answer synthesis
HISTORY_MODEL = os.getenv("TLI_HISTORY_MODEL", "claude-haiku-4-5")  # bulk history gen
PROFILE_MODEL = os.getenv("TLI_PROFILE_MODEL", HISTORY_MODEL)      # per-book profiles

# --- Embeddings ---
EMBED_BACKEND = os.getenv("TLI_EMBED_BACKEND", "local").lower()
LOCAL_EMBED_MODEL = os.getenv("TLI_LOCAL_EMBED_MODEL", "intfloat/multilingual-e5-large")
VOYAGE_EMBED_MODEL = os.getenv("TLI_VOYAGE_EMBED_MODEL", "voyage-3")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def ensure_dirs() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
