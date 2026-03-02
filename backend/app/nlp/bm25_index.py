# app/nlp/bm25_index.py
from __future__ import annotations

from typing import List, Dict, Tuple

import joblib
import pandas as pd
from rank_bm25 import BM25Okapi

from app.config import get_course_paths
from app.ingestion.preprocess import tokenize


def build_bm25_index(
    all_chunks: List[Dict],
    prof_id: str,
    course_id: str,
) -> Tuple[BM25Okapi, pd.DataFrame]:
    """
    Build a BM25 index for a specific (prof_id, course_id) and save artifacts.
    """
    paths = get_course_paths(prof_id, course_id)
    df = pd.DataFrame(all_chunks)

    corpus_text = (
        df["chunk_text"].fillna("")
        + " "
        + df.get("chunk_summary", "").fillna("")
        + " "
        + df.get("chunk_questions", "").fillna("")
    )
    tokenized_corpus = corpus_text.apply(tokenize).tolist()
    bm25 = BM25Okapi(tokenized_corpus)

    paths["index_dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(bm25, paths["bm25_model"])
    return bm25, df
