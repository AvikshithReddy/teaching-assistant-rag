# app/eval/evaluation.py
from __future__ import annotations

from typing import Dict, List
import argparse
import time

import pandas as pd

from app.retrieval.retriever import retrieve_hybrid


def _parse_gold_ids(row: pd.Series) -> List[str]:
    raw = str(row.get("gold_chunk_ids", "") or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_gold_docs(row: pd.Series) -> List[str]:
    raw = str(row.get("gold_doc_names", "") or "").strip()
    if not raw:
        doc = str(row.get("gold_doc_name", "") or "").strip()
        return [doc] if doc else []
    return [x.strip() for x in raw.split(",") if x.strip()]


def evaluate_retrieval(df: pd.DataFrame, top_k: int = 8) -> Dict[str, float]:
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    latencies = []

    for _, row in df.iterrows():
        question = str(row.get("question", "") or "")
        prof_id = str(row.get("prof_id", "") or "")
        course_id = str(row.get("course_id", "") or "")

        gold_chunk_ids = set(_parse_gold_ids(row))
        gold_docs = set(_parse_gold_docs(row))

        t0 = time.time()
        retrieved = retrieve_hybrid(question, prof_id, course_id, top_k=top_k)
        latencies.append(time.time() - t0)

        if not retrieved:
            precision_scores.append(0.0)
            recall_scores.append(0.0)
            mrr_scores.append(0.0)
            continue

        hits = 0
        first_rank = None
        for i, r in enumerate(retrieved, start=1):
            hit = False
            if gold_chunk_ids and r.get("chunk_id") in gold_chunk_ids:
                hit = True
            if gold_docs and r.get("doc_name") in gold_docs:
                hit = True
            if hit:
                hits += 1
                if first_rank is None:
                    first_rank = i

        precision_scores.append(hits / max(len(retrieved), 1))
        if gold_chunk_ids or gold_docs:
            denom = max(len(gold_chunk_ids) or len(gold_docs), 1)
            recall_scores.append(hits / denom)
        else:
            recall_scores.append(0.0)

        mrr_scores.append(1.0 / first_rank if first_rank else 0.0)

    return {
        "precision_at_k": float(sum(precision_scores) / max(len(precision_scores), 1)),
        "recall_at_k": float(sum(recall_scores) / max(len(recall_scores), 1)),
        "mrr": float(sum(mrr_scores) / max(len(mrr_scores), 1)),
        "avg_latency_s": float(sum(latencies) / max(len(latencies), 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--csv", required=True, help="CSV with question, prof_id, course_id, gold_* columns")
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    metrics = evaluate_retrieval(df, top_k=args.top_k)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
