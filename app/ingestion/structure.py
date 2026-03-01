# app/ingestion/structure.py
from __future__ import annotations

from typing import Dict, List
import re


def _clean_line(line: str) -> str:
    return (line or "").rstrip()


def _is_heading(line: str) -> bool:
    if not line:
        return False
    if len(line) > 100:
        return False
    if re.match(r"^\d+(\.\d+)*\s+\S+", line):
        return True
    # ALL CAPS or Title Case-ish and short
    letters = re.sub(r"[^A-Za-z]", "", line)
    if letters and letters.isupper():
        return True
    if line.istitle() and len(line.split()) <= 8:
        return True
    return False


def _is_table_line(line: str) -> bool:
    if "|" in line and line.count("|") >= 2:
        return True
    # multiple columns separated by >=2 spaces or tabs
    if re.search(r"\S+\s{2,}\S+", line):
        return True
    return False


def _is_code_line(line: str) -> bool:
    if line.startswith("```"):
        return True
    if re.match(r"^\s{4,}\S+", line):
        return True
    if re.search(r"[{}();=<>]", line) and re.search(r"\b(def|class|return|for|if|else)\b", line):
        return True
    return False


def _is_list_line(line: str) -> bool:
    return bool(re.match(r"^\s*([-*•]|\d+\.)\s+\S+", line))


def parse_text_to_blocks(text: str) -> List[Dict]:
    """
    Heuristically split text into structural blocks:
      heading, paragraph, table, code, list
    """
    lines = [_clean_line(l) for l in (text or "").splitlines()]
    blocks: List[Dict] = []

    buf: List[str] = []
    buf_type: str | None = None

    def flush():
        nonlocal buf, buf_type
        if buf:
            blocks.append({"block_type": buf_type or "paragraph", "text": "\n".join(buf).strip()})
        buf = []
        buf_type = None

    for line in lines:
        if not line.strip():
            flush()
            continue

        if _is_heading(line):
            flush()
            blocks.append({"block_type": "heading", "text": line.strip()})
            continue

        if _is_table_line(line):
            if buf_type not in (None, "table"):
                flush()
            buf_type = "table"
            buf.append(line)
            continue

        if _is_code_line(line):
            if buf_type not in (None, "code"):
                flush()
            buf_type = "code"
            buf.append(line)
            continue

        if _is_list_line(line):
            if buf_type not in (None, "list"):
                flush()
            buf_type = "list"
            buf.append(line)
            continue

        if buf_type not in (None, "paragraph"):
            flush()
        buf_type = "paragraph"
        buf.append(line)

    flush()
    return [b for b in blocks if b.get("text")]

