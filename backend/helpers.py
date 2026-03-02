"""
Non-Streamlit helper functions extracted from main_app.py.
Routers import from here to avoid pulling in the streamlit dependency.
"""
import datetime as dt
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas.errors import EmptyDataError

from app.config import (
    DATA_DIR,
    DATA_RAW_DIR,
    COURSES_CSV_PATH,
    CHAT_LOGS_CSV_PATH,
    get_course_index_dir,
    get_course_paths,
    _safe_id,
    RAG_STORAGE_BACKEND,
)
from app.ingestion.pdf_loader import load_pdf
from app.ingestion.pptx_loader import load_pptx
from app.ingestion.structure import parse_text_to_blocks
from app.ingestion.chunking import chunk_blocks
from app.nlp.tfidf_index import build_tfidf_index
from app.nlp.bm25_index import build_bm25_index
from app.nlp.embeddings_index import build_embeddings_index

MATERIALS_CSV_PATH = DATA_DIR / "materials.csv"


# ---------- Courses ----------

def _load_courses() -> pd.DataFrame:
    if COURSES_CSV_PATH.exists() and COURSES_CSV_PATH.stat().st_size > 0:
        try:
            return pd.read_csv(COURSES_CSV_PATH)
        except EmptyDataError:
            pass
    return pd.DataFrame(columns=["prof_id", "prof_name", "course_id", "course_title"])


def _save_courses(df: pd.DataFrame) -> None:
    COURSES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COURSES_CSV_PATH, index=False)


def upsert_course(prof_id: str, prof_name: str, course_id: str, course_title: str) -> None:
    df = _load_courses()
    prof_id = str(prof_id).strip()
    course_id = str(course_id).strip()
    row = {
        "prof_id": prof_id,
        "prof_name": str(prof_name).strip(),
        "course_id": course_id,
        "course_title": str(course_title).strip(),
    }
    if df.empty:
        df = pd.DataFrame([row])
    else:
        mask = (df["prof_id"].astype(str) == prof_id) & (df["course_id"].astype(str) == course_id)
        if mask.any():
            df.loc[mask, ["prof_name", "course_title"]] = [row["prof_name"], row["course_title"]]
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save_courses(df)


def _rm_tree(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.rglob("*"), reverse=True):
        if p.is_file():
            p.unlink(missing_ok=True)
        else:
            p.rmdir()
    path.rmdir()


def clear_course_index(prof_id: str, course_id: str) -> None:
    idx_dir = get_course_index_dir(prof_id, course_id)
    if idx_dir.exists():
        _rm_tree(idx_dir)


def delete_course(prof_id: str, course_id: str) -> None:
    df = _load_courses()
    if not df.empty:
        mask = ~(
            (df["prof_id"].astype(str) == str(prof_id))
            & (df["course_id"].astype(str) == str(course_id))
        )
        df = df.loc[mask].copy()
        _save_courses(df)

    clear_course_index(prof_id, course_id)

    raw_dir = get_course_raw_dir(prof_id, course_id)
    if raw_dir.exists():
        _rm_tree(raw_dir)

    mdf = _load_materials()
    if not mdf.empty:
        mdf = mdf.loc[
            ~(
                (mdf["prof_id"].astype(str) == str(prof_id))
                & (mdf["course_id"].astype(str) == str(course_id))
            )
        ].copy()
        _save_materials(mdf)

    if RAG_STORAGE_BACKEND == "postgres":
        try:
            from app.storage.postgres_store import delete_course_chunks
            delete_course_chunks(prof_id=prof_id, course_id=course_id)
        except Exception:
            pass


# ---------- Materials ----------

def get_course_raw_dir(prof_id: str, course_id: str) -> Path:
    d = DATA_RAW_DIR / _safe_id(prof_id) / _safe_id(course_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_materials() -> pd.DataFrame:
    if MATERIALS_CSV_PATH.exists() and MATERIALS_CSV_PATH.stat().st_size > 0:
        try:
            return pd.read_csv(MATERIALS_CSV_PATH)
        except EmptyDataError:
            pass
    return pd.DataFrame(columns=["prof_id", "course_id", "filename", "stored_path", "uploaded_at"])


def _save_materials(df: pd.DataFrame) -> None:
    MATERIALS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MATERIALS_CSV_PATH, index=False)


def add_material_record(prof_id: str, course_id: str, filename: str, stored_path: Path) -> None:
    df = _load_materials()
    row = {
        "prof_id": str(prof_id),
        "course_id": str(course_id),
        "filename": filename,
        "stored_path": str(stored_path),
        "uploaded_at": dt.datetime.utcnow().isoformat(),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.drop_duplicates(subset=["prof_id", "course_id", "filename"], keep="last")
    _save_materials(df)


def list_materials(prof_id: str, course_id: str) -> pd.DataFrame:
    df = _load_materials()
    if df.empty:
        return df
    return df[
        (df["prof_id"].astype(str) == str(prof_id))
        & (df["course_id"].astype(str) == str(course_id))
    ].copy()


def delete_material(prof_id: str, course_id: str, filename: str) -> None:
    df = _load_materials()
    if df.empty:
        return
    hit = df[
        (df["prof_id"].astype(str) == str(prof_id))
        & (df["course_id"].astype(str) == str(course_id))
        & (df["filename"].astype(str) == str(filename))
    ]
    if not hit.empty:
        p = Path(hit.iloc[0]["stored_path"])
        if p.exists():
            p.unlink(missing_ok=True)
    df = df.loc[
        ~(
            (df["prof_id"].astype(str) == str(prof_id))
            & (df["course_id"].astype(str) == str(course_id))
            & (df["filename"].astype(str) == str(filename))
        )
    ].copy()
    _save_materials(df)


# ---------- Index ----------

def rebuild_index_for_course(prof_id: str, course_id: str) -> int:
    clear_course_index(prof_id, course_id)

    mats = list_materials(prof_id, course_id)
    if mats.empty:
        return 0

    all_pages: List[Dict] = []
    for _, r in mats.iterrows():
        p = Path(r["stored_path"])
        if not p.exists():
            continue
        doc_name = p.stem
        if p.suffix.lower() == ".pdf":
            pages = load_pdf(p, course_id=course_id, doc_name=doc_name)
        elif p.suffix.lower() == ".pptx":
            pages = load_pptx(p, course_id=course_id, doc_name=doc_name)
        else:
            continue
        all_pages.extend(pages)

    all_chunks: List[Dict] = []
    for item in all_pages:
        blocks = parse_text_to_blocks(item["text"])
        chunks = chunk_blocks(
            blocks=blocks,
            course_id=item["course_id"],
            doc_name=item["doc_name"],
            source_type=item["source_type"],
            page_or_slide=int(item["page_or_slide"]),
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    build_tfidf_index(all_chunks=all_chunks, prof_id=prof_id, course_id=course_id)
    build_bm25_index(all_chunks=all_chunks, prof_id=prof_id, course_id=course_id)
    _model, _index, embeddings, df_chunks = build_embeddings_index(
        prof_id=prof_id, course_id=course_id
    )

    if RAG_STORAGE_BACKEND == "postgres":
        try:
            from app.storage.postgres_store import upsert_chunks
            upsert_chunks(
                chunks=df_chunks.to_dict(orient="records"),
                embeddings=embeddings,
                prof_id=prof_id,
                course_id=course_id,
            )
        except Exception:
            pass

    return len(all_chunks)
