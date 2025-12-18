from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder

_reranker_model: Optional[CrossEncoder] = None

def get_reranker() -> CrossEncoder:
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model

def rerank(query: str, passages: List[Dict], top_k: int = 5) -> List[Dict]:
    if not passages:
        return []

    model = get_reranker()
    pairs = [(query, (p.get("chunk_text") or p.get("text") or "")) for p in passages]
    scores = model.predict(pairs)

    out = []
    for p, s in zip(passages, scores):
        p2 = dict(p)
        p2["rerank_score"] = float(s)
        out.append(p2)

    out.sort(key=lambda x: x["rerank_score"], reverse=True)
    return out[:top_k]