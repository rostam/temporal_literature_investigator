"""Phase 3 storage — a single Chroma collection holding both books and events.

We compute embeddings ourselves (see embeddings.py) and hand them to Chroma, so
the collection is created without an embedding function. Every chunk carries
metadata (type, year, decade, country) to enable temporal/geographic filtering
at query time.
"""
from __future__ import annotations

import chromadb

from . import config


def get_collection():
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(extra_metadata: dict | None = None):
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION)
    except Exception:  # noqa: BLE001 — fine if it didn't exist
        pass
    metadata = {"hnsw:space": "cosine"}
    if extra_metadata:
        metadata.update(extra_metadata)
    return client.get_or_create_collection(name=config.COLLECTION, metadata=metadata)
