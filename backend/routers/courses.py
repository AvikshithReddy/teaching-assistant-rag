import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from routers.auth import require_role, current_user
from helpers import _load_courses, upsert_course, delete_course

router = APIRouter(tags=["courses"])


class CourseBody(BaseModel):
    course_id: str
    course_title: str


# ---------- Professor ----------

@router.get("/professor/courses")
def list_professor_courses(prof: dict = Depends(require_role("professor"))):
    df = _load_courses()
    if df.empty:
        return []
    rows = df[df["prof_id"].astype(str) == str(prof["id"])].copy()
    return rows.to_dict(orient="records")


@router.post("/professor/courses", status_code=status.HTTP_201_CREATED)
def create_course(body: CourseBody, prof: dict = Depends(require_role("professor"))):
    if not body.course_id.strip() or not body.course_title.strip():
        raise HTTPException(status_code=400, detail="course_id and course_title are required")
    upsert_course(prof["id"], prof["name"], body.course_id.strip(), body.course_title.strip())
    return {"prof_id": prof["id"], "course_id": body.course_id.strip(), "course_title": body.course_title.strip()}


@router.delete("/professor/courses/{course_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_course(course_id: str, prof: dict = Depends(require_role("professor"))):
    delete_course(prof["id"], course_id)


# ---------- Student ----------

@router.get("/student/courses")
def list_all_courses(student: dict = Depends(require_role("student"))):
    df = _load_courses()
    if df.empty:
        return []
    return df.sort_values(["course_id", "prof_id"]).to_dict(orient="records")
