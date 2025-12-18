# app/retrieval/retriever.py
from __future__ import annotations

from typing import List, Dict, Tuple, Any
import re

import numpy as np
import pandas as pd
from scipy import sparse
import joblib
import faiss
from sentence_transformers import SentenceTransformer

# Streamlit cache is optional at import-time (so tests/CLI still work)
try:
    import streamlit as st
except Exception:  # noqa: BLE001
    st = None

from app.config import EMBEDDING_MODEL_NAME, get_course_paths
from app.ingestion.preprocess import preprocess_for_tfidf


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
def _load_tfidf_components_cached(prof_id: str, course_id: str):
    paths = get_course_paths(prof_id, course_id)

    if (
        not paths["tfidf_model"].exists()
        or not paths["tfidf_matrix"].exists()
        or not paths["chunks_csv"].exists()
    ):
        raise FileNotFoundError("TF-IDF index not built for this course.")

    vectorizer = joblib.load(paths["tfidf_model"])
    tfidf_matrix = sparse.load_npz(paths["tfidf_matrix"])
    df = pd.read_csv(paths["chunks_csv"])
    return vectorizer, tfidf_matrix, df


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


def _keyword_boost(df: pd.DataFrame, question: str) -> np.ndarray:
    if df.empty:
        return np.zeros(0, dtype="float32")

    text_series = df["chunk_text"].fillna("").str.lower()
    tokens = re.findall(r"\b\w+\b", question.lower())
    tokens = sorted({t for t in tokens if len(t) > 3})

    boost = np.zeros(len(df), dtype="float32")
    for kw in tokens:
        mask = text_series.str.contains(rf"\b{re.escape(kw)}\b", regex=True)
        if mask.any():
            boost[mask.to_numpy()] += 1.0

    if boost.max() > 0:
        boost = boost / (boost.max() + 1e-9) * 0.5
    return boost.astype("float32")


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
                    "course_id": row["course_id"],
                    "doc_name": row["doc_name"],
                    "source_type": row["source_type"],
                    "page_or_slide": int(row["page_or_slide"]),
                    "chunk_text": row["chunk_text"],
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
    tfidf_weight: float = 0.5,
    use_keyword_boost: bool = True,
) -> List[Dict]:
    try:
        vectorizer, tfidf_matrix, df_tfidf = _load_tfidf_components_cached(prof_id, course_id)
        df_emb, _embeddings, faiss_index = _load_embedding_components_cached(prof_id, course_id)
    except FileNotFoundError:
        return []

    if df_tfidf.empty:
        return []

    # TF-IDF
    pre_q = preprocess_for_tfidf(query)
    q_vec = vectorizer.transform([pre_q])
    tfidf_scores = (q_vec @ tfidf_matrix.T).toarray().flatten()

    # Embeddings (FAISS)
    model = _get_embedding_model()
    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    k_candidates = min(max(top_k * 3, top_k), len(df_emb))
    if k_candidates <= 0:
        return []

    scores, idxs = faiss_index.search(q_emb, k=k_candidates)
    emb_scores = np.zeros(len(df_emb), dtype="float32")
    for rank, idx in enumerate(idxs[0]):
        emb_scores[idx] = float(scores[0][rank])

    tfidf_norm = _safe_norm(tfidf_scores.astype("float32"))
    emb_norm = _safe_norm(emb_scores)

    hybrid_scores = tfidf_weight * tfidf_norm + (1.0 - tfidf_weight) * emb_norm
    if use_keyword_boost:
        hybrid_scores += _keyword_boost(df_tfidf, query)

    top_indices = np.argsort(hybrid_scores)[::-1]
    results: List[Dict] = []
    for idx in top_indices:
        score = float(hybrid_scores[idx])
        if score <= 0:
            continue
        row = df_tfidf.iloc[idx]
        results.append(
            {
                "score": score,
                "course_id": row["course_id"],
                "doc_name": row["doc_name"],
                "source_type": row["source_type"],
                "page_or_slide": int(row["page_or_slide"]),
                "chunk_text": row["chunk_text"],
            }
        )
        if len(results) >= top_k:
            break

    return results