"""Phase 3 — chunk books + events, embed, and load into the vector store.

Chunking strategy (deliberately simple for a prototype):
  - one "book" chunk per review (title + meta + character gist + body head)
  - one "event" chunk per historical event

Metadata on every chunk: type, year, decade, country (+ title/author/slug for books).
Chroma metadata values must be scalars, so None is dropped.
"""
from __future__ import annotations

from . import config
from .embeddings import get_embedder
from .history import load_events
from .ingest import load_books
from .store import reset_collection

_BODY_HEAD_CHARS = 1500  # keep book chunks focused; bump or split for production


def _meta(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _book_docs(books: list[dict]):
    ids, docs, metas = [], [], []
    for b in books:
        text = f"{b['summary_text']}\n\n{b['body'][:_BODY_HEAD_CHARS]}"
        ids.append(f"book::{b['slug']}")
        docs.append(text)
        metas.append(_meta({
            "type": "book",
            "title": b["title_fa"],
            "title_en": b["title_en"],
            "author": b["author"],
            "year": b["year"],
            "decade": b["decade"],
            "country": b["country"],
            "genre": b["genre"],
            "slug": b["slug"],
        }))
    return ids, docs, metas


def _event_docs(events: list[dict]):
    ids, docs, metas = [], [], []
    for i, e in enumerate(events):
        text = f"[{e['country']} {e['year']}] {e['title']} — {e['summary']}"
        ids.append(f"event::{e['country']}::{e['year']}::{i}")
        docs.append(text)
        metas.append(_meta({
            "type": "event",
            "title": e["title"],
            "year": e.get("year"),
            "decade": e.get("decade"),
            "country": e.get("country"),
            "kind": e.get("kind"),
        }))
    return ids, docs, metas


def _add_in_batches(coll, embedder, ids, docs, metas, batch=64):
    for i in range(0, len(ids), batch):
        sl = slice(i, i + batch)
        embeddings = embedder.embed_documents(docs[sl])
        coll.add(ids=ids[sl], documents=docs[sl],
                 metadatas=metas[sl], embeddings=embeddings)


def run() -> int:
    config.ensure_dirs()
    embedder = get_embedder()
    coll = reset_collection()  # idempotent full rebuild

    bi, bd, bm = _book_docs(load_books())
    ei, ed, em = _event_docs(load_events())

    _add_in_batches(coll, embedder, bi, bd, bm)
    _add_in_batches(coll, embedder, ei, ed, em)

    total = len(bi) + len(ei)
    print(f"Indexed {len(bi)} books + {len(ei)} events = {total} chunks "
          f"into '{config.COLLECTION}' ({config.EMBED_BACKEND} embeddings).")
    if not ei:
        print("  note: no events indexed — run `tli build-history` first for "
              "the history side of the join.")
    return total
