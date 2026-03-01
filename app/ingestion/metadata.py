# app/ingestion/metadata.py
from __future__ import annotations

from typing import List
import re
from textwrap import shorten

from app.ingestion.preprocess import STOP_WORDS


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,}")


def _sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def summarize_text(text: str, max_chars: int = 260) -> str:
    sents = _sentences(text)
    if not sents:
        return ""
    summary = " ".join(sents[:2])
    return shorten(summary, width=max_chars, placeholder=" …")


def extract_keywords(text: str, k: int = 8) -> List[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text or "")]
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    if not tokens:
        return []
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # sort by freq desc, then alpha
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]
    return [t for t, _ in top]


def generate_questions(section_title: str, keywords: List[str], max_q: int = 3) -> List[str]:
    out: List[str] = []
    clean_title = (section_title or "").strip()
    for kw in keywords:
        if clean_title:
            out.append(f"What is {kw} in the context of {clean_title}?")
        else:
            out.append(f"What is {kw}?")
        if len(out) >= max_q:
            break
    if not out and clean_title:
        out.append(f"What is {clean_title}?")
    return out[:max_q]

