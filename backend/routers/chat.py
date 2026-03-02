import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from routers.auth import require_role
from app.qa.rag_pipeline import answer_question
from app.qa.chat_memory import load_chat_history

router = APIRouter(tags=["chat"])


class AskBody(BaseModel):
    question: str
    prof_id: str
    top_k: int = 8


@router.get("/student/courses/{course_id}/history")
def chat_history(course_id: str, prof_id: str, student: dict = Depends(require_role("student"))):
    msgs = load_chat_history(prof_id=prof_id, course_id=course_id, student_id=student["id"], limit=20)
    return msgs


@router.post("/student/courses/{course_id}/ask")
async def ask(course_id: str, body: AskBody, student: dict = Depends(require_role("student"))):
    result = await asyncio.to_thread(
        answer_question,
        question=body.question,
        prof_id=body.prof_id,
        course_id=course_id,
        student_id=student["id"],
        student_name=student["name"],
        top_k=body.top_k,
    )
    return {"answer": result["answer"], "sources": result.get("sources", [])}
