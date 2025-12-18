# app/config.py

import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# -------------------------------
# Base directories
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_INDEX_ROOT = DATA_DIR / "index"

for d in (DATA_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_INDEX_ROOT):
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Model settings
# -------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80

# -------------------------------
# Safe ID handling
# -------------------------------
def _safe_id(value: Any) -> str:
    """Normalize any value (int/str/None) into a filesystem-safe string."""
    if value is None:
        return "unknown"
    value = str(value)
    return re.sub(r"[^a-zA-Z0-9_-]", "_", value)

# -------------------------------
# Paths for indexes
# -------------------------------
def get_course_index_dir(prof_id: Any, course_id: Any) -> Path:
    path = DATA_INDEX_ROOT / _safe_id(prof_id) / _safe_id(course_id)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_course_paths(prof_id: Any, course_id: Any) -> dict:
    base = get_course_index_dir(prof_id, course_id)
    return {
        "index_dir": base,
        "tfidf_model": base / "tfidf_vectorizer.pkl",
        "tfidf_matrix": base / "tfidf_matrix.npz",
        "chunks_csv": base / "chunks_metadata.csv",
        "embeddings_matrix": base / "embeddings.npy",
        "faiss_index": base / "faiss_index.bin",
    }




def get_student_chat_path(prof_id: Any, course_id: Any, student_id: Any) -> Path:
    base = DATA_DIR / "chat_logs" / _safe_id(prof_id) / _safe_id(course_id)
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_safe_id(student_id)}.csv"
# -------------------------------
# Metadata Files
# -------------------------------
COURSES_CSV_PATH = DATA_DIR / "courses.csv"
CHAT_LOGS_CSV_PATH = DATA_DIR / "chat_logs.csv"


