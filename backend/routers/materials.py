import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from routers.auth import require_role
from helpers import add_material_record, delete_material, get_course_raw_dir, list_materials

router = APIRouter(tags=["materials"])


@router.get("/professor/courses/{course_id}/materials")
def get_materials(course_id: str, prof: dict = Depends(require_role("professor"))):
    df = list_materials(prof["id"], course_id)
    if df.empty:
        return []
    return df.to_dict(orient="records")


@router.post("/professor/courses/{course_id}/materials")
async def upload_materials(
    course_id: str,
    files: list[UploadFile] = File(...),
    prof: dict = Depends(require_role("professor")),
):
    existing_df = list_materials(prof["id"], course_id)
    existing_names = set(existing_df["filename"].astype(str).tolist()) if not existing_df.empty else set()
    raw_dir = get_course_raw_dir(prof["id"], course_id)

    uploaded, skipped = 0, 0
    for uf in files:
        fname = uf.filename
        if fname in existing_names or (raw_dir / fname).exists():
            skipped += 1
            continue
        dest = raw_dir / fname
        content = await uf.read()
        dest.write_bytes(content)
        add_material_record(prof["id"], course_id, fname, dest)
        uploaded += 1

    return {"uploaded": uploaded, "skipped": skipped}


@router.delete("/professor/courses/{course_id}/materials/{filename}", status_code=status.HTTP_204_NO_CONTENT)
def remove_material(course_id: str, filename: str, prof: dict = Depends(require_role("professor"))):
    delete_material(prof["id"], course_id, filename)
