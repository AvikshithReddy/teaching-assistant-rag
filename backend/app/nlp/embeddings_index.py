# app/nlp/embeddings_index.py

from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

from app.config import EMBEDDING_MODEL_NAME, get_course_paths


def build_embeddings_index(
    prof_id: str,
    course_id: str,
) -> Tuple[SentenceTransformer, faiss.IndexFlatIP, np.ndarray, pd.DataFrame]:
    """
    Build a sentence-transformer embeddings index for (prof_id, course_id)
    and save FAISS + embeddings matrix to disk.
    """
    paths = get_course_paths(prof_id, course_id)
    df = pd.read_csv(paths["chunks_csv"])

    texts = df["chunk_text"].astype(str).tolist()

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    np.save(paths["embeddings_matrix"], embeddings)
    faiss.write_index(index, str(paths["faiss_index"]))

    return model, index, embeddings, df