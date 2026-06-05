"""Phase 4 — retrieval + Claude synthesis.

Pipeline:
  1. (optional) metadata pre-filter: year range and/or country
  2. vector similarity over the filtered set
  3. assemble book + event context with source tags
  4. Claude answers bilingually, citing the post slugs it used

Streaming + adaptive thinking on the synthesis model (Opus). The system prompt
is the stable prefix; only the per-question content varies, so prompt caching
kicks in across repeated calls.
"""
from __future__ import annotations

from . import config
from .embeddings import get_embedder
from .store import get_collection

_SYSTEM = (
    "You are the Temporal Literature Investigator. You explore how historical "
    "events shaped the novels of an era and vice versa, grounded ONLY in the "
    "retrieved context provided.\n"
    "Rules:\n"
    "- Answer in the SAME language as the user's question (Persian or English).\n"
    "- Draw explicit connections between BOOK context and HISTORICAL EVENT context.\n"
    "- Cite the book reviews you rely on by their [slug] tag.\n"
    "- If the context is insufficient to connect literature and history, say so "
    "plainly rather than inventing facts."
)


def _filter_clauses(year_min, year_max, country) -> list[dict]:
    clauses = []
    if year_min is not None:
        clauses.append({"year": {"$gte": int(year_min)}})
    if year_max is not None:
        clauses.append({"year": {"$lte": int(year_max)}})
    if country:
        clauses.append({"country": country})
    return clauses


def _where(clauses: list[dict]) -> dict | None:
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def retrieve(question: str, k: int = 9, *, year_min=None, year_max=None,
             country=None) -> list[dict]:
    """Balanced retrieval across chunk types, then merge.

    A history-phrased query otherwise lets one type crowd out the others, but the
    whole point is to join literature (books + profiles) with history (events) —
    so we guarantee all three are represented. Empty types (e.g. before profiles
    are built) simply contribute nothing.
    """
    coll = get_collection()
    qvec = get_embedder().embed_query(question)
    base = _filter_clauses(year_min, year_max, country)

    def query_type(chunk_type: str, n: int) -> list[dict]:
        if n <= 0:
            return []
        res = coll.query(query_embeddings=[qvec], n_results=n,
                         where=_where(base + [{"type": chunk_type}]),
                         include=["documents", "metadatas", "distances"])
        return [{"doc": d, "meta": m, "distance": dist}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0],
                                      res["distances"][0])]

    third = max(1, k // 3)
    return (query_type("book", third)
            + query_type("profile", third)
            + query_type("event", k - 2 * third))


def _format_context(hits: list[dict]) -> str:
    books, profiles, events = [], [], []
    for h in hits:
        m = h["meta"]
        t = m.get("type")
        if t == "book":
            tag = m.get("slug", "?")
            books.append(f"[{tag}] ({m.get('country')}, {m.get('year')}) "
                         f"{m.get('title')} — {m.get('author')}\n{h['doc']}")
        elif t == "profile":
            tag = m.get("slug", "?")
            profiles.append(f"[{tag}] {h['doc']}")
        else:
            events.append(f"({m.get('country')}, {m.get('year')}) {h['doc']}")
    parts = []
    if books:
        parts.append("=== BOOK REVIEWS ===\n" + "\n\n".join(books))
    if profiles:
        parts.append("=== BOOK PROFILES (setting / themes / plot) ===\n"
                     + "\n\n".join(profiles))
    if events:
        parts.append("=== HISTORICAL EVENTS ===\n" + "\n".join(events))
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def ask(question: str, *, k: int = 8, year_min=None, year_max=None,
        country=None, stream: bool = True) -> str:
    from .llm import get_llm

    hits = retrieve(question, k=k, year_min=year_min, year_max=year_max,
                    country=country)
    context = _format_context(hits)
    user = f"Context:\n{context}\n\n---\nQuestion: {question}"

    llm = get_llm()
    fn = llm.stream if stream else llm.complete
    return fn(_SYSTEM, user, model=config.SYNTH_MODEL, max_tokens=2000)
