# app/nlp/tfidf_index.py

from typing import List, Dict, Tuple
from pathlib import Path

import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from app.config import DATA_RAW_DIR, CHUNK_SIZE, CHUNK_OVERLAP, get_course_paths
from app.ingestion.preprocess import preprocess_for_tfidf


def save_raw_upload(file_bytes: bytes, filename: str) -> Path:
    """
    Save uploaded file to data/raw and return the saved path.
    Raw storage is global; indexing is per-course.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_RAW_DIR / filename
    with path.open("wb") as f:
        f.write(file_bytes)
    return path


def chunk_text(
    text: str,
    course_id: str,
    doc_name: str,
    source_type: str,
    page_or_slide: int,
) -> List[Dict]:
    """
    Split text into overlapping chunks.
    """
    words = text.split()
    chunks: List[Dict] = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        if not chunk_words:
            break

        chunk_text_str = " ".join(chunk_words)
        chunks.append(
            {
                "course_id": course_id,
                "doc_name": doc_name,
                "source_type": source_type,
                "page_or_slide": page_or_slide,
                "chunk_text": chunk_text_str,
            }
        )

        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP

    return chunks


def build_tfidf_index(
    all_chunks: List[Dict],
    prof_id: str,
    course_id: str,
) -> Tuple[TfidfVectorizer, sparse.csr_matrix, pd.DataFrame]:
    """
    Build a TF-IDF index for a specific (prof_id, course_id) and save artifacts.
    """
    paths = get_course_paths(prof_id, course_id)
    df = pd.DataFrame(all_chunks)

    processed_texts = df["chunk_text"].apply(preprocess_for_tfidf).tolist()

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    paths["index_dir"].mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, paths["tfidf_model"])
    sparse.save_npz(paths["tfidf_matrix"], tfidf_matrix)
    df.to_csv(paths["chunks_csv"], index=False)

    return vectorizer, tfidf_matrix, df