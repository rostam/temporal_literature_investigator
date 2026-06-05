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
from .profiles import load_profiles, setting_midpoint
from .store import reset_collection
from .utils import decade_of

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


def _profile_docs(books: list[dict], profiles: dict[str, dict]):
    by_slug = {b["slug"]: b for b in books}
    ids, docs, metas = [], [], []
    for slug, p in profiles.items():
        b = by_slug.get(slug)
        if not b:
            continue
        chars = "; ".join(f"{c.get('name')}: {c.get('role')}"
                          for c in p.get("characters", [])[:15])
        themes = ", ".join(p.get("themes", []))
        period = ""
        if p.get("setting_start"):
            period = f"{p['setting_start']}–{p.get('setting_end') or p['setting_start']}"
        text = (
            f"[PROFILE] {b['title_fa']} — {b['author']}\n"
            f"Set in: {p.get('setting_place', '?')} {period}\n"
            f"Themes: {themes}\n"
            f"Historical backdrop: {p.get('historical_backdrop', '')}\n"
            f"Plot: {p.get('plot_synopsis', '')}\n"
            f"Characters: {chars}"
        )
        # year = story-time midpoint so temporal filters match the setting,
        # not just the publication year (book chunk already carries publication).
        syear = setting_midpoint(p, b["year"])
        ids.append(f"profile::{slug}")
        docs.append(text)
        metas.append(_meta({
            "type": "profile",
            "title": b["title_fa"],
            "author": b["author"],
            "year": syear,
            "decade": decade_of(syear) if syear else None,
            "country": b["country"],            # author's country (filter-consistent)
            "setting_place": p.get("setting_place"),
            "themes": themes or None,
            "confidence": p.get("confidence"),
            "slug": slug,
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

    books = load_books()
    bi, bd, bm = _book_docs(books)
    pi, pd, pm = _profile_docs(books, load_profiles())
    ei, ed, em = _event_docs(load_events())

    _add_in_batches(coll, embedder, bi, bd, bm)
    _add_in_batches(coll, embedder, pi, pd, pm)
    _add_in_batches(coll, embedder, ei, ed, em)

    total = len(bi) + len(pi) + len(ei)
    print(f"Indexed {len(bi)} books + {len(pi)} profiles + {len(ei)} events "
          f"= {total} chunks into '{config.COLLECTION}' "
          f"({config.EMBED_BACKEND} embeddings).")
    if not pi:
        print("  note: no profiles indexed — run `tli build-profiles` for the "
              "richer plot/setting/theme layer.")
    if not ei:
        print("  note: no events indexed — run `tli build-history` first for "
              "the history side of the join.")
    return total
