"""Phase 1 — parse the Jekyll _posts/*.md into a clean books.jsonl.

Each record:
  {slug, date, title_fa, title_en, author, year, decade, country,
   genre, literary_school, rating, characters:[{name,desc}], body, summary_text}

The single most important transform here is normalizing the publication year,
because every temporal query depends on it.
"""
from __future__ import annotations

import json
import re

import frontmatter

from . import config
from .utils import clean, decade_of, normalize_year

# Body table is `| key | value |` rows in Persian.
_TABLE_KEYS = {
    "work_name": "نام اثر",
    "author": "نویسنده",
    "title_en": "نام اصلی اثر",
    "year": "سال چاپ",
    "country": "کشور",
    "genre": "ژانر",
    "literary_school": "مکتب ادبی",
    "rating": "امتیاز",
}

_CHAR_RE = re.compile(r"<summary>(.*?)</summary>(.*?)</details>", re.S)
_RATING_RE = re.compile(r"(\d{1,2})\s*/\s*10")


def _parse_table(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.split("|")]
        if len(cols) > 2 and cols[1]:
            out[cols[1]] = cols[2]
    return out


def _extract_characters(body: str) -> list[dict[str, str]]:
    chars = []
    for name, desc in _CHAR_RE.findall(body):
        chars.append({"name": clean(name), "desc": clean(desc)})
    return chars


def _rating_score(raw: str) -> int | None:
    m = _RATING_RE.search(raw or "")
    return int(m.group(1)) if m else None


def parse_post(path) -> dict | None:
    post = frontmatter.load(path)
    body = post.content
    table = _parse_table(body)

    def tcell(key: str) -> str:
        return table.get(_TABLE_KEYS[key], "")

    fm = post.metadata
    title_fa = str(fm.get("title", "")).strip()
    tags = fm.get("tags", []) or []
    categories = fm.get("categories", []) or []

    # Year: prefer the table cell, fall back to any year-looking tag.
    year = normalize_year(tcell("year"))
    if year is None:
        for t in tags:
            year = normalize_year(str(t))
            if year:
                break

    rating = _rating_score(tcell("rating"))
    if rating is None:
        rating = _rating_score(str(fm.get("description", "")) + " ".join(map(str, tags)))

    characters = _extract_characters(body)

    # A compact, embeddable summary: title + meta + character gist.
    char_gist = " | ".join(f"{c['name']}: {c['desc']}" for c in characters[:8])
    summary_text = clean(
        f"{title_fa}. نویسنده: {tcell('author')}. کشور: {tcell('country')}. "
        f"ژانر: {tcell('genre')}. سال: {year}. شخصیت‌ها: {char_gist}"
    )

    slug = path.stem  # e.g. 2023-06-01-The-Brothers-Karamazov-By-...
    date = "-".join(slug.split("-")[:3])

    return {
        "slug": slug,
        "date": date,
        "url_title": slug[11:],  # strip the YYYY-MM-DD- prefix
        "title_fa": title_fa or tcell("work_name"),
        "title_en": clean(tcell("title_en")),
        "author": tcell("author"),
        "year": year,
        "decade": decade_of(year) if year else None,
        "country": tcell("country"),
        "genre": tcell("genre"),
        "literary_school": tcell("literary_school"),
        "rating": rating,
        "categories": [str(c) for c in categories],
        "characters": characters,
        "summary_text": summary_text,
        "body": clean(body),
    }


def run() -> int:
    config.ensure_dirs()
    if not config.POSTS_DIR.is_dir():
        raise SystemExit(f"Posts dir not found: {config.POSTS_DIR}\n"
                         f"Set TLI_BOOKS_REPO in .env")

    records, skipped = [], 0
    for path in sorted(config.POSTS_DIR.glob("*.md")):
        rec = parse_post(path)
        if rec is None:
            continue
        if rec["year"] is None:
            skipped += 1  # kept, but flag it — temporal queries can't place it
        records.append(rec)

    with config.BOOKS_JSONL.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    placed = sum(1 for r in records if r["year"])
    print(f"Parsed {len(records)} posts -> {config.BOOKS_JSONL}")
    print(f"  with a normalized year: {placed}  |  without (year=None): {skipped}")
    return len(records)


def load_books() -> list[dict]:
    if not config.BOOKS_JSONL.exists():
        raise SystemExit("books.jsonl missing — run `tli ingest` first.")
    with config.BOOKS_JSONL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
