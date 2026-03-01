# app/retrieval/retriever.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import re

import numpy as np
import pandas as pd
import joblib
import faiss
from sentence_transformers import SentenceTransformer

# Streamlit cache is optional at import-time (so tests/CLI still work)
try:
    import streamlit as st
except Exception:  # noqa: BLE001
    st = None

from app.config import (
    EMBEDDING_MODEL_NAME,
    get_course_paths,
    FUSION_METHOD,
    RRF_K,
    BM25_WEIGHT,
    VECTOR_WEIGHT,
    MIN_VECTOR_SIM,
    MIN_BM25_SCORE,
    MIN_HYBRID_SCORE,
    RAG_STORAGE_BACKEND,
)
from app.ingestion.preprocess import tokenize
from app.retrieval.postgres_backend import retrieve_hybrid_postgres


# -----------------------------
# Streamlit caching wrappers
# -----------------------------

def _cache_resource(func):
    if st is None:
        return func
    return st.cache_resource(show_spinner=False)(func)


def _cache_data(func):
    if st is None:
        return func
    return st.cache_data(show_spinner=False)(func)


@_cache_resource
def _get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@_cache_data
def _load_bm25_components_cached(prof_id: str, course_id: str):
    paths = get_course_paths(prof_id, course_id)

    if (
        not paths["bm25_model"].exists()
        or not paths["chunks_csv"].exists()
    ):
        raise FileNotFoundError("BM25 index not built for this course.")

    bm25 = joblib.load(paths["bm25_model"])
    df = pd.read_csv(paths["chunks_csv"])
    return bm25, df


@_cache_data
def _load_embedding_components_cached(prof_id: str, course_id: str):
    paths = get_course_paths(prof_id, course_id)

    if (
        not paths["faiss_index"].exists()
        or not paths["embeddings_matrix"].exists()
        or not paths["chunks_csv"].exists()
    ):
        raise FileNotFoundError("Embeddings index not built for this course.")

    df = pd.read_csv(paths["chunks_csv"])
    embeddings = np.load(paths["embeddings_matrix"])
    index = faiss.read_index(str(paths["faiss_index"]))
    return df, embeddings, index


def _safe_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    m = float(x.max())
    if m == 0:
        return x
    return x / (m + 1e-9)


def _apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]]) -> np.ndarray:
    if df.empty or not filters:
        return np.ones(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    for k, v in filters.items():
        if k not in df.columns:
            continue
        if isinstance(v, (list, tuple, set)):
            mask &= df[k].astype(str).isin([str(x) for x in v]).to_numpy()
        else:
            mask &= (df[k].astype(str) == str(v)).to_numpy()
    return mask


def _rrf_scores(order: np.ndarray, k: int) -> np.ndarray:
    scores = np.zeros(len(order), dtype="float32")
    for rank, idx in enumerate(order, start=1):
        scores[idx] = 1.0 / (k + rank)
    return scores


def keyword_fallback(prof_id: str, course_id: str, question: str, max_hits: int = 5) -> List[Dict]:
    paths = get_course_paths(prof_id, course_id)
    chunks_path = paths["chunks_csv"]
    if not chunks_path.exists():
        return []

    df = pd.read_csv(chunks_path)
    if df.empty:
        return []

    text_series = df["chunk_text"].fillna("").str.lower()
    tokens = re.findall(r"\b\w+\b", question.lower())
    tokens = sorted({t for t in tokens if len(t) > 3})

    hits: List[Dict] = []
    for kw in tokens:
        mask = text_series.str.contains(r"\b" + re.escape(kw) + r"\b", regex=True)
        matches = df[mask].head(3)
        for _, row in matches.iterrows():
            hits.append(
                {
                    "score": 1.0,
                    "chunk_id": row.get("chunk_id", ""),
                    "course_id": row.get("course_id", ""),
                    "doc_name": row.get("doc_name", ""),
                    "source_type": row.get("source_type", ""),
                    "page_or_slide": int(row.get("page_or_slide", 0)),
                    "section_title": row.get("section_title", ""),
                    "chunk_text": row.get("chunk_text", ""),
                    "chunk_summary": row.get("chunk_summary", ""),
                    "chunk_keywords": row.get("chunk_keywords", ""),
                    "chunk_questions": row.get("chunk_questions", ""),
                }
            )

    seen = set()
    deduped: List[Dict] = []
    for h in hits:
        key = (h["doc_name"], h["page_or_slide"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
        if len(deduped) >= max_hits:
            break
    return deduped


def retrieve_hybrid(
    query: str,
    prof_id: str,
    course_id: str,
    top_k: int = 8,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    # Optional Postgres backend
    if RAG_STORAGE_BACKEND == "postgres":
        try:
            model = _get_embedding_model()
            q_emb = model.encode([query], normalize_embeddings=True)
            q_emb = np.asarray(q_emb, dtype="float32")[0]
            results = retrieve_hybrid_postgres(
                query=query,
                query_emb=q_emb,
                prof_id=prof_id,
                course_id=course_id,
                top_k=top_k,
            )
            if results:
                return results
        except Exception:
            pass

    try:
        bm25, df = _load_bm25_components_cached(prof_id, course_id)
        df_emb, _embeddings, faiss_index = _load_embedding_components_cached(prof_id, course_id)
    except FileNotFoundError:
        return []

    if df.empty:
        return []

    # BM25
    tokens = tokenize(query)
    bm25_scores = np.asarray(bm25.get_scores(tokens), dtype="float32")

    # Embeddings (FAISS)
    model = _get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    k_candidates = min(max(top_k * 5, top_k), len(df_emb))
    if k_candidates <= 0:
        return []

    scores, idxs = faiss_index.search(q_emb, k=k_candidates)
    emb_scores = np.zeros(len(df_emb), dtype="float32")
    for rank, idx in enumerate(idxs[0]):
        emb_scores[idx] = float(scores[0][rank])

    # Apply basic thresholds
    bm25_scores[bm25_scores < MIN_BM25_SCORE] = 0.0
    emb_scores[emb_scores < MIN_VECTOR_SIM] = 0.0

    # Apply filters
    mask = _apply_filters(df, filters)
    bm25_scores = bm25_scores * mask
    emb_scores = emb_scores * mask

    if FUSION_METHOD == "rrf":
        bm25_order = np.argsort(bm25_scores)[::-1]
        emb_order = np.argsort(emb_scores)[::-1]
        rrf_bm25 = _rrf_scores(bm25_order, RRF_K)
        rrf_emb = _rrf_scores(emb_order, RRF_K)
        hybrid_scores = BM25_WEIGHT * rrf_bm25 + VECTOR_WEIGHT * rrf_emb
    else:
        bm25_norm = _safe_norm(bm25_scores)
        emb_norm = _safe_norm(emb_scores)
        hybrid_scores = BM25_WEIGHT * bm25_norm + VECTOR_WEIGHT * emb_norm

    top_indices = np.argsort(hybrid_scores)[::-1]
    results: List[Dict] = []
    for idx in top_indices:
        score = float(hybrid_scores[idx])
        if score < MIN_HYBRID_SCORE:
            continue
        row = df.iloc[idx]
        results.append(
            {
                "score": score,
                "bm25_score": float(bm25_scores[idx]),
                "vector_sim": float(emb_scores[idx]),
                "chunk_id": row.get("chunk_id", ""),
                "course_id": row.get("course_id", ""),
                "doc_name": row.get("doc_name", ""),
                "source_type": row.get("source_type", ""),
                "page_or_slide": int(row.get("page_or_slide", 0)),
                "section_title": row.get("section_title", ""),
                "chunk_text": row.get("chunk_text", ""),
                "chunk_summary": row.get("chunk_summary", ""),
                "chunk_keywords": row.get("chunk_keywords", ""),
                "chunk_questions": row.get("chunk_questions", ""),
                "chunk_len_tokens": int(row.get("chunk_len_tokens", 0) or 0),
            }
        )
        if len(results) >= top_k:
            break

    return results
