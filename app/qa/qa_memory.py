# app/qa/qa_memory.py

from __future__ import annotations

import json
import datetime as dt
from functools import lru_cache
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import CHAT_LOGS_CSV_PATH, EMBEDDING_MODEL_NAME


# ---------- Helpers to load / save chat log ----------

def _load_chat_df() -> pd.DataFrame:
    """Return full chat log (all students / courses)."""
    if CHAT_LOGS_PATH.exists():
        return pd.read_csv(CHAT_LOGS_PATH)
    return pd.DataFrame(
        columns=[
            "timestamp",
            "student_id",
            "course_id",
            "question",
            "answer",
            "sources_json",
        ]
    )


def _save_chat_df(df: pd.DataFrame) -> None:
    CHAT_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CHAT_LOGS_PATH, index=False)


# ---------- Public: logging ----------

def log_interaction(
    *,
    student_id: Optional[str],
    course_id: Optional[str],
    question: str,
    answer: str,
    sources: List[Dict],
) -> None:
    """
    Append a single Q&A turn to the global chat log CSV.

    If student_id or course_id is missing (e.g., professor testing), we skip logging.
    """
    if not student_id or not course_id:
        return

    df = _load_chat_df()

    ts = dt.datetime.utcnow().isoformat()
    row = {
        "timestamp": ts,
        "student_id": str(student_id),
        "course_id": str(course_id),
        "question": question,
        "answer": answer,
        "sources_json": json.dumps(sources, ensure_ascii=False),
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_chat_df(df)


# ---------- Public: semantic retrieval over past Q&A ----------

@lru_cache(maxsize=1)
def _get_st_model() -> SentenceTransformer:
    """
    Cache the SentenceTransformer model across Streamlit reruns.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_relevant_qa_snippets(
    question: str,
    course_id: Optional[str],
    top_k: int = 3,
) -> List[str]:
    """
    Return up to top_k short snippets from past Q&A for this course,
    selected via cosine similarity between the current question and
    previous (question + answer) pairs.

    This makes the assistant "learn" from prior chats.
    """
    if not course_id:
        return []

    df = _load_chat_df()
    if df.empty:
        return []

    df_course = df[df["course_id"] == str(course_id)]
    if df_course.empty:
        return []

    model = _get_st_model()

    # Embed the current question
    q_emb = model.encode([question], normalize_embeddings=True)[0]  # shape (d,)

    # Embed all QA texts for this course
    qa_texts = (df_course["question"] + " " + df_course["answer"]).tolist()
    qa_embs = model.encode(qa_texts, normalize_embeddings=True)  # (N, d)

    # Cosine similarity = dot product because vectors are normalized
    scores = np.dot(qa_embs, q_emb)  # (N,)

    # Pick best indices
    top_k = min(top_k, len(scores))
    if top_k <= 0:
        return []

    top_idx = np.argsort(scores)[::-1][:top_k]

    snippets: List[str] = []
    for i in top_idx:
        row = df_course.iloc[i]
        snippet = (
            f"[Past Q&A – {row['timestamp']}]\n"
            f"Question: {row['question']}\n"
            f"Answer: {row['answer']}"
        )
        snippets.append(snippet)

    return snippets



