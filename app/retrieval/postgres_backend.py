# app/retrieval/postgres_backend.py
from __future__ import annotations

from typing import Dict, List

import numpy as np

from app.config import BM25_WEIGHT, VECTOR_WEIGHT, MIN_VECTOR_SIM, MIN_BM25_SCORE, PG_SCHEMA
from app.storage.postgres_store import get_conn, HAS_PG


def retrieve_hybrid_postgres(
    *,
    query: str,
    query_emb: np.ndarray,
    prof_id: str,
    course_id: str,
    top_k: int,
) -> List[Dict]:
    if not HAS_PG:
        return []

    # cosine distance -> similarity
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT
              chunk_id, doc_name, source_type, page_or_slide, section_title,
              chunk_text, chunk_summary, chunk_keywords, chunk_questions, chunk_len_tokens,
              ts_rank_cd(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) AS bm25_score,
              1 - (embedding <=> %s) AS vector_sim
            FROM {PG_SCHEMA}.rag_chunks
            WHERE prof_id = %s AND course_id = %s
            ORDER BY (ts_rank_cd(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) * %s
                 + (1 - (embedding <=> %s)) * %s) DESC
            LIMIT %s;
            """,
            (
                query,
                query_emb.tolist(),
                str(prof_id),
                str(course_id),
                query,
                BM25_WEIGHT,
                query_emb.tolist(),
                VECTOR_WEIGHT,
                int(top_k),
            ),
        )
        rows = cur.fetchall()

    results: List[Dict] = []
    for r in rows:
        bm25_score = float(r[10] or 0.0)
        vector_sim = float(r[11] or 0.0)
        if bm25_score < MIN_BM25_SCORE and vector_sim < MIN_VECTOR_SIM:
            continue
        results.append(
            {
                "score": (bm25_score * BM25_WEIGHT + vector_sim * VECTOR_WEIGHT),
                "chunk_id": str(r[0]),
                "doc_name": r[1],
                "source_type": r[2],
                "page_or_slide": int(r[3] or 0),
                "section_title": r[4],
                "chunk_text": r[5],
                "chunk_summary": r[6],
                "chunk_keywords": r[7],
                "chunk_questions": r[8],
                "chunk_len_tokens": int(r[9] or 0),
                "bm25_score": bm25_score,
                "vector_sim": vector_sim,
            }
        )
    return results

