# app/storage/postgres_store.py
from __future__ import annotations

from typing import Iterable, List, Dict

import numpy as np

from app.config import PG_DSN, PG_SCHEMA

try:
    import psycopg
    from pgvector.psycopg import register_vector
    HAS_PG = True
except Exception:  # noqa: BLE001
    HAS_PG = False


def _require_pg() -> None:
    if not HAS_PG:
        raise RuntimeError("Postgres dependencies not installed. Add psycopg[binary] and pgvector.")
    if not PG_DSN:
        raise RuntimeError("PG_DSN is not set for Postgres backend.")


def get_conn():
    _require_pg()
    conn = psycopg.connect(PG_DSN)
    register_vector(conn)
    return conn


def ensure_schema(conn, embedding_dim: int) -> None:
    cur = conn.cursor()
    cur.execute(f"CREATE SCHEMA IF NOT EXISTS {PG_SCHEMA};")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {PG_SCHEMA}.rag_chunks (
            chunk_id UUID PRIMARY KEY,
            prof_id TEXT NOT NULL,
            course_id TEXT NOT NULL,
            doc_name TEXT,
            source_type TEXT,
            page_or_slide INT,
            section_title TEXT,
            chunk_text TEXT,
            chunk_summary TEXT,
            chunk_keywords TEXT,
            chunk_questions TEXT,
            chunk_len_tokens INT,
            embedding VECTOR({embedding_dim})
        );
        """
    )
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS rag_chunks_prof_course_idx
        ON {PG_SCHEMA}.rag_chunks (prof_id, course_id);
        """
    )
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS rag_chunks_text_idx
        ON {PG_SCHEMA}.rag_chunks
        USING GIN (to_tsvector('english', chunk_text));
        """
    )
    cur.execute(
        f"""
        CREATE INDEX IF NOT EXISTS rag_chunks_vec_idx
        ON {PG_SCHEMA}.rag_chunks
        USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
        """
    )
    conn.commit()


def upsert_chunks(
    *,
    chunks: Iterable[Dict],
    embeddings: np.ndarray,
    prof_id: str,
    course_id: str,
) -> None:
    _require_pg()
    if embeddings is None or len(embeddings) == 0:
        return

    emb_dim = int(embeddings.shape[1])
    with get_conn() as conn:
        ensure_schema(conn, emb_dim)
        cur = conn.cursor()

        rows: List[tuple] = []
        for i, ch in enumerate(chunks):
            rows.append(
                (
                    ch["chunk_id"],
                    prof_id,
                    course_id,
                    ch.get("doc_name"),
                    ch.get("source_type"),
                    int(ch.get("page_or_slide") or 0),
                    ch.get("section_title"),
                    ch.get("chunk_text"),
                    ch.get("chunk_summary"),
                    ch.get("chunk_keywords"),
                    ch.get("chunk_questions"),
                    int(ch.get("chunk_len_tokens") or 0),
                    embeddings[i].tolist(),
                )
            )

        cur.executemany(
            f"""
            INSERT INTO {PG_SCHEMA}.rag_chunks
            (chunk_id, prof_id, course_id, doc_name, source_type, page_or_slide, section_title,
             chunk_text, chunk_summary, chunk_keywords, chunk_questions, chunk_len_tokens, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (chunk_id) DO UPDATE SET
              prof_id = EXCLUDED.prof_id,
              course_id = EXCLUDED.course_id,
              doc_name = EXCLUDED.doc_name,
              source_type = EXCLUDED.source_type,
              page_or_slide = EXCLUDED.page_or_slide,
              section_title = EXCLUDED.section_title,
              chunk_text = EXCLUDED.chunk_text,
              chunk_summary = EXCLUDED.chunk_summary,
              chunk_keywords = EXCLUDED.chunk_keywords,
              chunk_questions = EXCLUDED.chunk_questions,
              chunk_len_tokens = EXCLUDED.chunk_len_tokens,
              embedding = EXCLUDED.embedding;
            """,
            rows,
        )
        conn.commit()


def delete_course_chunks(*, prof_id: str, course_id: str) -> None:
    if not HAS_PG or not PG_DSN:
        return
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"DELETE FROM {PG_SCHEMA}.rag_chunks WHERE prof_id = %s AND course_id = %s;",
            (str(prof_id), str(course_id)),
        )
        conn.commit()

