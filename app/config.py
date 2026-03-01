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
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Approximate token-based chunking (word count proxy)
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "420"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "70"))
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "520"))

# Retrieval / fusion
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf")  # "rrf" or "weighted"
RRF_K = int(os.getenv("RRF_K", "60"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.45"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.55"))

# Similarity thresholds / gating
MIN_VECTOR_SIM = float(os.getenv("MIN_VECTOR_SIM", "0.20"))
MIN_BM25_SCORE = float(os.getenv("MIN_BM25_SCORE", "0.10"))
MIN_HYBRID_SCORE = float(os.getenv("MIN_HYBRID_SCORE", "0.10"))

# Storage backend: "local" or "postgres"
RAG_STORAGE_BACKEND = os.getenv("RAG_STORAGE_BACKEND", "local").lower()
PG_DSN = os.getenv("PG_DSN", "")
PG_SCHEMA = os.getenv("PG_SCHEMA", "rag")

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
        "bm25_model": base / "bm25.pkl",
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
