import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends

from routers.auth import require_role
from helpers import rebuild_index_for_course
from app.config import get_course_paths

router = APIRouter(tags=["index"])


@router.post("/professor/courses/{course_id}/rebuild-index", status_code=202)
async def rebuild_index(course_id: str, prof: dict = Depends(require_role("professor"))):
    asyncio.create_task(asyncio.to_thread(rebuild_index_for_course, prof["id"], course_id))
    return {"status": "started"}


@router.get("/professor/courses/{course_id}/index-status")
def index_status(course_id: str, prof: dict = Depends(require_role("professor"))):
    paths = get_course_paths(prof["id"], course_id)
    chunks_csv: Path = paths["chunks_csv"]
    if not chunks_csv.exists() or chunks_csv.stat().st_size == 0:
        return {"indexed": False, "chunk_count": 0}
    try:
        import pandas as pd
        df = pd.read_csv(chunks_csv)
        return {"indexed": True, "chunk_count": len(df)}
    except Exception:
        return {"indexed": False, "chunk_count": 0}
