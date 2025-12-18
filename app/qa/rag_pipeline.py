# app/qa/rag_pipeline.py
from __future__ import annotations

from typing import List, Dict, Optional
from textwrap import shorten

from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL_NAME
from app.retrieval.retriever import retrieve_hybrid, keyword_fallback
from app.qa.chat_memory import load_chat_history, append_chat_history

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


def build_context_snippet(retrieved_chunks: List[Dict]) -> str:
    blocks = []
    for i, ch in enumerate(retrieved_chunks, start=1):
        header = (
            f"[Source {i} – {ch['doc_name']} – {ch['source_type']} "
            f"– page/slide {ch['page_or_slide']}]"
        )
        body = shorten(str(ch.get("chunk_text", "")), width=950, placeholder=" …")
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks)


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
    Professor/student ready RAG:
      1) Hybrid retrieval (TF-IDF + embeddings + keyword boost)
      2) Keyword fallback if weak/empty
      3) Optional cross-encoder rerank
      4) LLM answer using context + student chat history
      5) Persist chat per student per course
    """
    client = _get_client()

    # ---- Retrieval ----
    retrieved = retrieve_hybrid(question, prof_id, course_id, top_k=top_k)

    if not retrieved:
        retrieved = keyword_fallback(prof_id, course_id, question, max_hits=top_k)
    else:
        max_score = max((r.get("score", 0.0) for r in retrieved), default=0.0)
        if max_score < 0.05:
            fallback_hits = keyword_fallback(prof_id, course_id, question, max_hits=top_k)
            if fallback_hits:
                retrieved = fallback_hits

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

    append_chat_history(
        prof_id=prof_id,
        course_id=course_id,
        student_id=student_id,
        student_name=student_name,
        question=question,
        answer=answer_text,
    )

    return {"answer": answer_text, "sources": retrieved}