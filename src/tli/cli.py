"""Command-line entry point.

  tli ingest                      parse the Jekyll posts -> data/books.jsonl
  tli build-history [--force]     generate the historical-events layer (Claude)
  tli index                       embed books + events into the vector store
  tli ask "..." [--country .. --from 1914 --to 1918 -k 8 --no-stream]
  tli info                        show corpus / store stats

Typical first run:  tli ingest && tli build-history && tli index && tli ask "..."
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    p = argparse.ArgumentParser(prog="tli", description="Temporal Literature Investigator")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="parse posts into books.jsonl")

    ph = sub.add_parser("build-history", help="generate historical-events layer")
    ph.add_argument("--force", action="store_true", help="regenerate cached timelines")

    pp = sub.add_parser("build-profiles",
                        help="generate per-book literary-historical profiles")
    pp.add_argument("--force", action="store_true", help="regenerate cached profiles")

    sub.add_parser("index", help="embed + load the vector store")
    sub.add_parser("info", help="show stats")

    pa = sub.add_parser("ask", help="ask a question")
    pa.add_argument("question")
    pa.add_argument("-k", type=int, default=9, help="chunks to retrieve")
    pa.add_argument("--country", default=None)
    pa.add_argument("--from", dest="year_min", type=int, default=None)
    pa.add_argument("--to", dest="year_max", type=int, default=None)
    pa.add_argument("--no-stream", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        from . import ingest
        ingest.run()
    elif args.cmd == "build-history":
        from . import history
        history.run(force=args.force)
    elif args.cmd == "build-profiles":
        from . import profiles
        profiles.run(force=args.force)
    elif args.cmd == "index":
        from . import index
        index.run()
    elif args.cmd == "info":
        _info()
    elif args.cmd == "ask":
        from . import rag
        rag.ask(args.question, k=args.k, year_min=args.year_min,
                year_max=args.year_max, country=args.country,
                stream=not args.no_stream)
    return 0


def _info() -> None:
    from . import config
    from .ingest import load_books

    print(f"books repo : {config.BOOKS_REPO}")
    print(f"llm backend: {config.LLM_BACKEND}")
    print(f"embeddings : {config.EMBED_BACKEND}")
    print(f"synth model: {config.SYNTH_MODEL}   history model: {config.HISTORY_MODEL}")
    if config.BOOKS_JSONL.exists():
        books = load_books()
        placed = sum(1 for b in books if b["year"])
        countries = sorted({b["country"] for b in books if b["country"]})
        print(f"books.jsonl: {len(books)} records ({placed} with a year)")
        print(f"countries  : {len(countries)} -> {', '.join(countries[:12])}"
              + (" ..." if len(countries) > 12 else ""))
    else:
        print("books.jsonl: (not built — run `tli ingest`)")
    n_hist = len(list(config.HISTORY_DIR.glob('*.json'))) if config.HISTORY_DIR.exists() else 0
    print(f"history    : {n_hist} (country, decade) timelines cached")
    n_prof = len(list(config.PROFILES_DIR.glob('*.json'))) if config.PROFILES_DIR.exists() else 0
    print(f"profiles   : {n_prof} book profiles cached")
    if config.CHROMA_DIR.exists():
        try:
            from .store import get_collection
            print(f"vector idx : {get_collection().count()} chunks")
        except Exception as e:  # noqa: BLE001
            print(f"vector idx : (error: {e})")


if __name__ == "__main__":
    raise SystemExit(main())
