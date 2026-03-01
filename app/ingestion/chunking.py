# app/ingestion/chunking.py
from __future__ import annotations

from typing import Dict, List
from uuid import uuid4

from app.config import CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, MAX_CHUNK_TOKENS
from app.ingestion.metadata import summarize_text, extract_keywords, generate_questions


def _token_count(text: str) -> int:
    return len((text or "").split())


def _join_blocks(blocks: List[Dict]) -> str:
    parts: List[str] = []
    for b in blocks:
        bt = b.get("block_type", "paragraph")
        text = b.get("text", "")
        if bt == "table":
            parts.append("[Table]\n" + text)
        elif bt == "code":
            parts.append("[Code]\n" + text)
        elif bt == "list":
            parts.append("[List]\n" + text)
        else:
            parts.append(text)
    return "\n\n".join(p for p in parts if p.strip())


def chunk_blocks(
    *,
    blocks: List[Dict],
    course_id: str,
    doc_name: str,
    source_type: str,
    page_or_slide: int,
) -> List[Dict]:
    """
    Structure-aware chunking:
      - headings remain attached to their content
      - tables/codes are kept intact as blocks
      - chunk size is approx token count (word count proxy)
    """
    chunks: List[Dict] = []
    cur_blocks: List[Dict] = []
    cur_tokens = 0
    cur_section = ""
    cur_block_types: List[str] = []

    def emit(blocks_to_emit: List[Dict], section_title: str, block_types: List[str]) -> None:
        text = _join_blocks(blocks_to_emit).strip()
        if not text:
            return
        tokens = _token_count(text)
        keywords = extract_keywords(text)
        summary = summarize_text(text)
        questions = generate_questions(section_title, keywords)
        chunks.append(
            {
                "chunk_id": str(uuid4()),
                "course_id": course_id,
                "doc_name": doc_name,
                "source_type": source_type,
                "page_or_slide": int(page_or_slide),
                "section_title": section_title,
                "block_types": ",".join(sorted(set(block_types))),
                "chunk_text": text,
                "chunk_summary": summary,
                "chunk_keywords": ", ".join(keywords),
                "chunk_questions": " | ".join(questions),
                "chunk_len_tokens": tokens,
            }
        )

    for b in blocks:
        bt = b.get("block_type", "paragraph")
        text = b.get("text", "").strip()
        if not text:
            continue

        if bt == "heading":
            if cur_blocks:
                emit(cur_blocks, cur_section, cur_block_types)
                cur_blocks = []
                cur_tokens = 0
                cur_block_types = []
            cur_section = text
            cur_blocks = [{"block_type": "heading", "text": text}]
            cur_block_types = ["heading"]
            cur_tokens = _token_count(text)
            continue

        block_tokens = _token_count(text)

        # Always keep tables/codes intact
        if bt in {"table", "code"} and block_tokens > MAX_CHUNK_TOKENS:
            # flush current
            if cur_blocks:
                emit(cur_blocks, cur_section, cur_block_types)
                cur_blocks = []
                cur_tokens = 0
                cur_block_types = []
            emit([b], cur_section, [bt])
            continue

        if cur_tokens + block_tokens > CHUNK_SIZE_TOKENS and cur_blocks:
            emit(cur_blocks, cur_section, cur_block_types)

            # overlap by tokens using last block text
            if CHUNK_OVERLAP_TOKENS > 0:
                overlap_text = _join_blocks(cur_blocks)[-CHUNK_OVERLAP_TOKENS * 6 :]
                if overlap_text:
                    cur_blocks = [{"block_type": "paragraph", "text": overlap_text}]
                    cur_tokens = _token_count(overlap_text)
                    cur_block_types = ["overlap"]
                else:
                    cur_blocks = []
                    cur_tokens = 0
                    cur_block_types = []
            else:
                cur_blocks = []
                cur_tokens = 0
                cur_block_types = []

        cur_blocks.append(b)
        cur_block_types.append(bt)
        cur_tokens += block_tokens

    if cur_blocks:
        emit(cur_blocks, cur_section, cur_block_types)

    return chunks
