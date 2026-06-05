"""Small shared helpers — mostly the all-important year/digit normalization."""
from __future__ import annotations

import re

# Persian (۰-۹) and Arabic-Indic (٠-٩) digits -> ASCII.
_DIGIT_MAP = {ord(p): str(i) for i, p in enumerate("۰۱۲۳۴۵۶۷۸۹")}
_DIGIT_MAP.update({ord(a): str(i) for i, a in enumerate("٠١٢٣٤٥٦٧٨٩")})


def to_ascii_digits(s: str) -> str:
    return (s or "").translate(_DIGIT_MAP)


def normalize_year(raw: str) -> int | None:
    """Pull a 3-4 digit Gregorian year out of a messy string.

    Handles Persian digits, ranges ('۱۸۸۰-۱۸۸۱' -> 1880), and stray text.
    Returns None when no plausible year is present.
    """
    digits = to_ascii_digits(raw or "")
    matches = re.findall(r"\d{3,4}", digits)
    for m in matches:
        y = int(m)
        if 800 <= y <= 2100:  # plausible publication-year window
            return y
    return None


def decade_of(year: int) -> int:
    return (year // 10) * 10


def clean(text: str) -> str:
    """Collapse whitespace; strip a few Jekyll/Markdown artifacts for embedding."""
    text = re.sub(r"<sup[^>]*>.*?</sup>", " ", text or "", flags=re.S)
    text = re.sub(r"\{%.*?%\}", " ", text, flags=re.S)  # liquid tags e.g. rating include
    text = re.sub(r"\s+", " ", text)
    return text.strip()
