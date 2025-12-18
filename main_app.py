import datetime as dt
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError

from app.config import (
    DATA_DIR,
    DATA_RAW_DIR,
    COURSES_CSV_PATH,
    CHAT_LOGS_CSV_PATH,
    get_course_index_dir,
    get_course_paths,
    _safe_id,
)

from app.ingestion.pdf_loader import load_pdf
from app.ingestion.pptx_loader import load_pptx
from app.nlp.tfidf_index import chunk_text, build_tfidf_index
from app.nlp.embeddings_index import build_embeddings_index
from app.qa.rag_pipeline import answer_question
from app.qa.chat_memory import load_chat_history  # ✅ needed for student portal


# -----------------------------
# Storage tracker for uploads
# -----------------------------
MATERIALS_CSV_PATH = DATA_DIR / "materials.csv"


# -----------------------------
# Helpers: Courses CRUD
# -----------------------------
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
    """
    Removes ALL index artifacts for a course (tfidf, faiss, embeddings, chunks csv).
    Prevents stale 'Total chunks indexed' when there are no files.
    """
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

    # delete index dir
    clear_course_index(prof_id, course_id)

    # delete uploaded raw materials folder
    raw_dir = get_course_raw_dir(prof_id, course_id)
    if raw_dir.exists():
        _rm_tree(raw_dir)

    # remove materials from tracker
    mdf = _load_materials()
    if not mdf.empty:
        mdf = mdf.loc[
            ~(
                (mdf["prof_id"].astype(str) == str(prof_id))
                & (mdf["course_id"].astype(str) == str(course_id))
            )
        ].copy()
        _save_materials(mdf)


# -----------------------------
# Helpers: Materials tracking + per-course raw folder
# -----------------------------
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

    # de-dupe same file name for same course (keep latest)
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


# -----------------------------
# Chat log append (global CSV)
# -----------------------------
def append_chat_log(
    prof_id: str,
    course_id: str,
    student_id: str,
    student_name: str,
    question: str,
    answer: str,
) -> None:
    CHAT_LOGS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": dt.datetime.utcnow().isoformat(),
        "prof_id": str(prof_id),
        "course_id": str(course_id),
        "student_id": str(student_id),
        "student_name": str(student_name),
        "question": question,
        "answer": answer,
    }

    if CHAT_LOGS_CSV_PATH.exists() and CHAT_LOGS_CSV_PATH.stat().st_size > 0:
        try:
            df = pd.read_csv(CHAT_LOGS_CSV_PATH)
        except EmptyDataError:
            df = pd.DataFrame(columns=row.keys())
    else:
        df = pd.DataFrame(columns=row.keys())

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CHAT_LOGS_CSV_PATH, index=False)


# -----------------------------
# Index build
# -----------------------------
def rebuild_index_for_course(prof_id: str, course_id: str) -> int:
    # ✅ clear old index first (prevents stale samples)
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
        chunks = chunk_text(
            text=item["text"],
            course_id=item["course_id"],
            doc_name=item["doc_name"],
            source_type=item["source_type"],
            page_or_slide=int(item["page_or_slide"]),
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    build_tfidf_index(all_chunks=all_chunks, prof_id=prof_id, course_id=course_id)
    build_embeddings_index(prof_id=prof_id, course_id=course_id)
    return len(all_chunks)


# -----------------------------
# UI: Session + Login
# -----------------------------
def reset_session() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]


def login_screen() -> None:
    st.title("📚 AI Teaching Assistant (Advanced RAG System)")
    st.caption("Login to continue")

    role = st.radio("I am a:", ["Professor", "Student"], horizontal=True)

    if role == "Professor":
        prof_id = st.text_input("Professor ID", value=st.session_state.get("prof_id", ""))
        prof_name = st.text_input("Professor Name", value=st.session_state.get("prof_name", ""))

        if st.button("Login as Professor", type="primary"):
            if not prof_id.strip() or not prof_name.strip():
                st.error("Please enter both Professor ID and Name.")
                return
            st.session_state.update(
                {
                    "role": "professor",
                    "prof_id": prof_id.strip(),
                    "prof_name": prof_name.strip(),
                    "logged_in": True,
                }
            )
            st.rerun()

    else:
        student_id = st.text_input("Student ID", value=st.session_state.get("student_id", ""))
        student_name = st.text_input("Student Name", value=st.session_state.get("student_name", ""))

        if st.button("Login as Student", type="primary"):
            if not student_id.strip() or not student_name.strip():
                st.error("Please enter both Student ID and Name.")
                return
            st.session_state.update(
                {
                    "role": "student",
                    "student_id": student_id.strip(),
                    "student_name": student_name.strip(),
                    "logged_in": True,
                }
            )
            st.rerun()


# -----------------------------
# Professor portal
# -----------------------------
def professor_portal() -> None:
    prof_id = st.session_state["prof_id"]
    prof_name = st.session_state["prof_name"]

    st.header("Professor Portal")
    st.write(f"Welcome, **{prof_name}** 👋")

    if st.button("Logout"):
        reset_session()
        st.rerun()

    st.divider()

    courses = _load_courses()
    my_courses = courses[courses["prof_id"].astype(str) == str(prof_id)].copy()

    st.subheader("Your Courses")
    options = ["(Create new course)"]
    if not my_courses.empty:
        options += [
            f"{r['course_id']} — {r['course_title']}"
            for _, r in my_courses.sort_values("course_id").iterrows()
        ]

    selected = st.selectbox("Select a course to manage:", options)

    st.subheader("Create / Update Course")
    if selected != "(Create new course)":
        sel_course_id = selected.split("—")[0].strip()
        row = my_courses[my_courses["course_id"].astype(str) == sel_course_id].iloc[0]
        course_id = st.text_input("Course ID (e.g., CS101, ML2024)", value=str(row["course_id"]))
        course_title = st.text_input("Course Title", value=str(row["course_title"]))
    else:
        course_id = st.text_input("Course ID (e.g., CS101, ML2024)", value="")
        course_title = st.text_input("Course Title", value="")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Save Course", type="primary"):
            if not course_id.strip() or not course_title.strip():
                st.error("Course ID and Course Title are required.")
            else:
                upsert_course(prof_id, prof_name, course_id.strip(), course_title.strip())
                st.success("Saved.")
                st.rerun()

    with col_b:
        if selected != "(Create new course)":
            if st.button("Delete This Course", type="secondary"):
                delete_course(prof_id, course_id.strip())
                st.success("Deleted course + materials + index.")
                st.rerun()

    if not course_id.strip():
        st.info("Create a course first, then upload materials and build the index.")
        return

    st.divider()
    st.subheader(f"Manage Materials for Course: {course_id.strip()}")

    # Load materials once (duplicate detection + UI)
    mats = list_materials(prof_id, course_id)
    existing_names = set(mats["filename"].astype(str).tolist()) if not mats.empty else set()

    uploaded = st.file_uploader(
        "Upload course materials (PDF, PPTX). You can select multiple files.",
        type=["pdf", "pptx"],
        accept_multiple_files=True,
    )

    if uploaded:
        raw_dir = get_course_raw_dir(prof_id, course_id)
        uploaded_count, skipped_count = 0, 0

        for uf in uploaded:
            fname = uf.name

            if fname in existing_names:
                st.warning(f"Already uploaded: **{fname}** (skipping).")
                skipped_count += 1
                continue

            dest = raw_dir / fname
            if dest.exists():
                st.warning(f"File already exists on disk: **{fname}** (skipping).")
                skipped_count += 1
                continue

            dest.write_bytes(uf.getbuffer())
            add_material_record(prof_id, course_id, fname, dest)
            uploaded_count += 1

        if uploaded_count:
            st.success(f"Uploaded {uploaded_count} file(s).")
        if skipped_count and not uploaded_count:
            st.info("No new files were uploaded.")

        mats = list_materials(prof_id, course_id)

    if st.button("Build / Rebuild Index"):
        with st.spinner("Building index..."):
            n_chunks = rebuild_index_for_course(prof_id, course_id)
        st.success(f"Done. Total chunks indexed: {n_chunks}")

    st.subheader("Uploaded Materials (per course)")
    if mats.empty:
        st.caption("No uploaded files yet.")
    else:
        mats_sorted = mats.sort_values("uploaded_at", ascending=False).reset_index(drop=True)
        for i, r in mats_sorted.iterrows():
            c1, c2, c3 = st.columns([6, 2, 2])
            fname = str(r["filename"])
            ts = str(r.get("uploaded_at", ""))[:19]

            with c1:
                st.write(f"**{fname}**")
            with c2:
                st.caption(ts)

            # ✅ unique key even if filenames repeat
            del_key = f"del::{_safe_id(prof_id)}::{_safe_id(course_id)}::{i}::{ts}::{fname}"
            with c3:
                if st.button("Delete", key=del_key):
                    delete_material(prof_id, course_id, fname)
                    st.success("Deleted file.")
                    st.rerun()

    st.subheader("Index status & sample")
    paths = get_course_paths(prof_id, course_id)
    chunks_csv = paths["chunks_csv"]
    if chunks_csv.exists() and chunks_csv.stat().st_size > 0:
        try:
            df_chunks = pd.read_csv(chunks_csv)
            st.write(f"Total chunks indexed: **{len(df_chunks)}**")
            st.dataframe(df_chunks.head(10), use_container_width=True)
        except EmptyDataError:
            st.warning("chunks_metadata.csv is empty. Rebuild index.")
    else:
        st.caption("Index not built yet for this course.")


# -----------------------------
# Student portal
# -----------------------------
def student_portal() -> None:
    student_id = st.session_state["student_id"]
    student_name = st.session_state["student_name"]

    st.header("Student Portal")
    st.write(f"Welcome, **{student_name}** 👋")

    if st.button("Logout"):
        reset_session()
        st.rerun()

    st.divider()

    courses = _load_courses()
    if courses.empty:
        st.warning("No courses available yet. Ask your professor to create a course and upload materials.")
        return

    courses = courses.sort_values(["course_id", "prof_name"])

    display = [
        f"{r['course_id']} — {r['course_title']} (Prof. {r['prof_name']} | {r['prof_id']})"
        for _, r in courses.iterrows()
    ]
    selected = st.selectbox("Select your course:", display)

    course_id = selected.split("—")[0].strip()
    prof_id = selected.split("|")[-1].replace(")", "").strip()

    # Separate chat per student + course + professor
    chat_key = f"{student_id}::{prof_id}::{course_id}"

    if st.session_state.get("active_chat_key") != chat_key:
        st.session_state["active_chat_key"] = chat_key
        st.session_state["chat"] = load_chat_history(
            prof_id=prof_id, course_id=course_id, student_id=student_id, limit=20
        )

    st.divider()
    st.subheader("Ask questions")

    for m in st.session_state.get("chat", []):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask a question from your course materials...")
    if not user_q:
        return

    st.session_state["chat"].append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            out = answer_question(
                question=user_q,
                prof_id=prof_id,
                course_id=course_id,
                student_id=student_id,
                student_name=student_name,
                top_k=8,
            )
            ans = out["answer"]
            st.markdown(ans)

            if out.get("sources"):
                with st.expander("Sources used", expanded=False):
                    for s in out["sources"]:
                        st.write(
                            f"- **{s['doc_name']}** ({s['source_type']}) "
                            f"page/slide **{s['page_or_slide']}** | score={s.get('score', 0):.3f}"
                        )

    st.session_state["chat"].append({"role": "assistant", "content": ans})

    # Optional global log (separate from per-student memory)
    append_chat_log(
        prof_id=prof_id,
        course_id=course_id,
        student_id=student_id,
        student_name=student_name,
        question=user_q,
        answer=ans,
    )


def main() -> None:
    st.set_page_config(page_title="AI Teaching Assistant (RAG)", layout="wide")

    if not st.session_state.get("logged_in"):
        login_screen()
        return

    role = st.session_state.get("role")
    if role == "professor":
        professor_portal()
    elif role == "student":
        student_portal()
    else:
        reset_session()
        login_screen()


if __name__ == "__main__":
    main()