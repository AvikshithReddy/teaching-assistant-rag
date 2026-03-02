# app/qa/rag_pipeline.py
from __future__ import annotations

from typing import List, Dict, Optional
from textwrap import shorten
import re

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL_NAME, MIN_HYBRID_SCORE
from app.retrieval.retriever import retrieve_hybrid, keyword_fallback
from app.qa.chat_memory import load_chat_history, append_chat_history
from app.ingestion.preprocess import STOP_WORDS

# Optional cross-encoder reranker
try:
    from app.nlp.reranker import rerank as cross_rerank
    HAS_RERANK = True
except Exception:
    HAS_RERANK = False


SYSTEM_PROMPT = """
You are a helpful teaching assistant for a specific university course.

Rules:
- Answer ONLY using the provided course materials (context) and the student's recent chat history.
- If the answer is not clearly present in the context, say you are not sure and suggest asking the professor.
- Be concise and clear, and reference where the answer comes from (slide/page).
- If the user asks for "names only", respond with names only.
""".strip()


_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set. Add OPENAI_API_KEY=... to your .env")
    _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def _decompose_question(question: str) -> List[str]:
    q = (question or "").strip()
    if not q:
        return []

    parts = re.split(r"[;\n]", q)
    subqs: List[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "?" in p:
            subqs.extend([x.strip() for x in p.split("?") if x.strip()])
        else:
            subqs.append(p)

    # Heuristic split for long multi-asks
    if len(subqs) == 1 and " and " in q and len(q) > 140:
        subqs = [x.strip() for x in q.split(" and ") if x.strip()]

    # cap to 3 to avoid explosion
    return subqs[:3]


def _merge_results(all_results: List[List[Dict]], top_k: int) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for res in all_results:
        for r in res:
            key = r.get("chunk_id") or f"{r.get('doc_name')}::{r.get('page_or_slide')}"
            if key not in merged or r.get("score", 0) > merged[key].get("score", 0):
                merged[key] = r
    out = list(merged.values())
    out.sort(key=lambda x: x.get("score", 0), reverse=True)
    return out[:top_k]


def build_context_snippet(retrieved_chunks: List[Dict]) -> str:
    blocks = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        section = ch.get("section_title") or ""
        header = (
            f"[Source {i} – {ch.get('doc_name','')} – {ch.get('source_type','')} "
            f"– page/slide {ch.get('page_or_slide','')}]"
        )
        if section:
            header += f" – section: {section}"
        body = shorten(str(ch.get("chunk_text", "")), width=950, placeholder=" …")
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


def _grounding_score(answer: str, context: str) -> float:
    ans_tokens = re.findall(r"[A-Za-z]{3,}", (answer or "").lower())
    ans_tokens = [t for t in ans_tokens if t not in STOP_WORDS]
    if not ans_tokens:
        return 0.0

    ctx_tokens = set(re.findall(r"[A-Za-z]{3,}", (context or "").lower()))
    ctx_tokens = {t for t in ctx_tokens if t not in STOP_WORDS}
    if not ctx_tokens:
        return 0.0

    overlap = sum(1 for t in ans_tokens if t in ctx_tokens)
    return overlap / max(len(ans_tokens), 1)


def _strategist_ok(answer: str, sub_questions: List[str]) -> bool:
    if not sub_questions:
        return True
    ans_tokens = set(re.findall(r"[A-Za-z]{3,}", (answer or "").lower()))
    ans_tokens = {t for t in ans_tokens if t not in STOP_WORDS}
    if not ans_tokens:
        return False
    for q in sub_questions:
        q_tokens = set(re.findall(r"[A-Za-z]{3,}", (q or "").lower()))
        q_tokens = {t for t in q_tokens if t not in STOP_WORDS}
        if q_tokens and not (q_tokens & ans_tokens):
            return False
    return True


def answer_question(
    *,
    question: str,
    prof_id: str,
    course_id: str,
    student_id: str,
    student_name: Optional[str] = None,
    top_k: int = 8,
) -> Dict:
    """
    Production-ready RAG:
      1) Planner: decompose multi-part questions
      2) Hybrid retrieval (BM25 + embeddings + fusion)
      3) Optional cross-encoder rerank
      4) LLM answer using context + student chat history
      5) Validation: gatekeeper + grounding audit
      6) Persist chat per student per course
    """
    client = _get_client()

    # ---- Retrieval ----
    subqs = _decompose_question(question)
    if not subqs:
        subqs = [question]

    all_results: List[List[Dict]] = []
    for q in subqs:
        all_results.append(retrieve_hybrid(q, prof_id, course_id, top_k=top_k))

    retrieved = _merge_results(all_results, top_k=top_k)

    if not retrieved:
        retrieved = keyword_fallback(prof_id, course_id, question, max_hits=top_k)

    if not retrieved:
        answer_text = (
            "I could not find relevant information in the course materials for this question. "
            "Please ask your professor."
        )
        append_chat_history(
            prof_id=prof_id,
            course_id=course_id,
            student_id=student_id,
            student_name=student_name,
            question=question,
            answer=answer_text,
        )
        return {"answer": answer_text, "sources": []}

    # Gatekeeper threshold
    max_score = max((r.get("score", 0.0) for r in retrieved), default=0.0)
    if max_score < MIN_HYBRID_SCORE:
        answer_text = (
            "I could not find enough support in the course materials to answer confidently. "
            "Please ask your professor."
        )
        append_chat_history(
            prof_id=prof_id,
            course_id=course_id,
            student_id=student_id,
            student_name=student_name,
            question=question,
            answer=answer_text,
        )
        return {"answer": answer_text, "sources": []}

    # Optional rerank (expects chunk_text)
    if HAS_RERANK and len(retrieved) > 1:
        retrieved = cross_rerank(question, retrieved, top_k=min(top_k, len(retrieved)))

    retrieved = retrieved[:top_k]
    context = build_context_snippet(retrieved)

    # Short-term memory: last few turns for THIS student + course + professor
    chat_history = load_chat_history(
        prof_id=prof_id,
        course_id=course_id,
        student_id=student_id,
        limit=6,
    )

    user_msg = (
        f"Course ID: {course_id}\n\n"
        f"Context from course materials:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using ONLY the context above. "
        "Cite sources as (Source # - doc - page/slide). "
        "If not clearly supported, say you're not sure and suggest asking the professor."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_msg})

    completion = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=messages,
        temperature=0.2,
        max_tokens=650,
    )

    answer_text = completion.choices[0].message.content.strip()

    # ---- Validation Layer ----
    has_citations = "(Source" in answer_text
    grounding = _grounding_score(answer_text, context)
    strategist_ok = _strategist_ok(answer_text, subqs)

    if not has_citations or grounding < 0.2 or not strategist_ok:
        answer_text = (
            "I could not verify this answer strongly enough from the provided materials. "
            "Please ask your professor."
        )

    append_chat_history(
        prof_id=prof_id,
        course_id=course_id,
        student_id=student_id,
        student_name=student_name,
        question=question,
        answer=answer_text,
    )

    return {"answer": answer_text, "sources": retrieved}
