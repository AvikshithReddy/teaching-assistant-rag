# app/ingestion/pdf_loader.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

# Fallback
from PyPDF2 import PdfReader

# Optional: better layout + images
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# Optional OCR for images/diagrams
try:
    import pytesseract
    from PIL import Image
    import io
    HAS_OCR = True
except Exception:
    HAS_OCR = False


def _clean(s: str) -> str:
    return (s or "").strip()


def _ocr_image_bytes(img_bytes: bytes) -> str:
    if not HAS_OCR or not img_bytes:
        return ""
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        text = pytesseract.image_to_string(im) or ""
        return _clean(text)
    except Exception:
        return ""


def load_pdf(path: Path, course_id: str, doc_name: str) -> List[Dict]:
    """
    Extracts:
      - page text (layout-aware if PyMuPDF is available)
      - image OCR text (if OCR available)
    Returns items with keys:
      course_id, doc_name, source_type, page_or_slide, text
    """
    chunks: List[Dict] = []

    if HAS_PYMUPDF:
        doc = fitz.open(str(path))
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_no = page_idx + 1

            # Better than PyPDF2 in many PDFs
            txt = _clean(page.get_text("text"))
            if txt:
                chunks.append(
                    {
                        "course_id": course_id,
                        "doc_name": doc_name,
                        "source_type": "pdf",
                        "page_or_slide": page_no,
                        "text": txt,
                    }
                )

            # Extract images → OCR (diagrams/charts often contain text labels)
            # Keep OCR as separate “figure” chunk inside the same page
            image_list = page.get_images(full=True) or []
            for img_i, img in enumerate(image_list):
                xref = img[0]
                try:
                    base = doc.extract_image(xref)
                    img_bytes = base.get("image", b"")
                except Exception:
                    continue

                ocr_txt = _ocr_image_bytes(img_bytes)
                if ocr_txt:
                    chunks.append(
                        {
                            "course_id": course_id,
                            "doc_name": doc_name,
                            "source_type": "pdf_figure_ocr",
                            "page_or_slide": page_no,
                            "text": f"[Figure OCR {img_i+1}]\n{ocr_txt}",
                        }
                    )

        return chunks

    # ---------- Fallback: PyPDF2 ----------
    reader = PdfReader(str(path))
    for page_idx, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if not text:
            continue
        chunks.append(
            {
                "course_id": course_id,
                "doc_name": doc_name,
                "source_type": "pdf",
                "page_or_slide": page_idx,
                "text": text,
            }
        )
    return chunks