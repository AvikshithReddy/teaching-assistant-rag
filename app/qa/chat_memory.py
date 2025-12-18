# app/qa/chat_memory.py
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from app.config import DATA_DIR, _safe_id


def get_chat_log_path(prof_id: str, course_id: str, student_id: str) -> Path:
    """
    Per-student, per-course chat log:
      data/chat_logs/<prof_id>/<course_id>/<student_id>/chat.csv
    """
    base = (
        DATA_DIR
        / "chat_logs"
        / _safe_id(prof_id)
        / _safe_id(course_id)
        / _safe_id(student_id)
    )
    base.mkdir(parents=True, exist_ok=True)
    return base / "chat.csv"


def load_chat_history(
    prof_id: str,
    course_id: str,
    student_id: str,
    limit: int = 20,
) -> List[Dict]:
    """
    Returns Streamlit/OpenAI style messages:
      [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]
    """
    path = get_chat_log_path(prof_id, course_id, student_id)
    if not path.exists() or path.stat().st_size == 0:
        return []

    try:
        df = pd.read_csv(path)
    except Exception:
        return []

    if df.empty:
        return []

    # newest -> oldest, then reverse back to chronological
    df = df.sort_values("timestamp", ascending=False).head(limit).iloc[::-1]

    messages: List[Dict] = []
    for _, row in df.iterrows():
        q = str(row.get("question", "") or "")
        a = str(row.get("answer", "") or "")
        if q.strip():
            messages.append({"role": "user", "content": q})
        if a.strip():
            messages.append({"role": "assistant", "content": a})

    return messages


def append_chat_history(
    prof_id: str,
    course_id: str,
    student_id: str,
    student_name: Optional[str],
    question: str,
    answer: str,
) -> None:
    path = get_chat_log_path(prof_id, course_id, student_id)

    row = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "prof_id": str(prof_id),
        "course_id": str(course_id),
        "student_id": str(student_id),
        "student_name": "" if student_name is None else str(student_name),
        "question": question,
        "answer": answer,
    }

    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.DataFrame(columns=row.keys())
    else:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)