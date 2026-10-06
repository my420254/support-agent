"""检索评测：加载 golden set → 检索 → 算 Hit@k / MRR。

用法：
    python -m eval.evaluate                    # dense 基线
    python -m eval.evaluate --retriever hybrid # dense + BM25 + RRF 混合
"""

import argparse
import json
import sys
from pathlib import Path

from agents.tools import search_knowledge_base
from eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank, unique_doc_ids

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "golden" / "golden.jsonl"


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l]


def build_search(retriever: str):
    """返回一个 search(query, top_k) -> list[dict] 的检索函数。"""
    if retriever in ("hybrid", "bm25", "rerank"):
        from retrieval.hybrid import HybridRetriever

        hr = HybridRetriever()
        if retriever == "hybrid":
            return hr.search
        if retriever == "bm25":
            return hr.search_bm25
        return hr.search_rerank
    return search_knowledge_base


def evaluate(search=None, top_k: int = 10, limit: int | None = None) -> dict:
    search = search or search_knowledge_base
    golden = load_golden()
    if limit:
        golden = golden[:limit]
    hits = {k: 0 for k in (1, 3, 5, 10)}
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for g in golden:
        results = search(g["question"], top_k=top_k)
        doc_ids = unique_doc_ids(results)
        for k in hits:
            hits[k] += hit_at_k(g["gt_doc_id"], doc_ids, k)
        mrr_sum += reciprocal_rank(g["gt_doc_id"], doc_ids)
        ndcg_sum += ndcg_at_k(g["gt_doc_id"], doc_ids, 5)

    n = len(golden)
    return {
        "n": n,
        "Hit@1": hits[1] / n if n else 0,
        "Hit@3": hits[3] / n if n else 0,
        "Hit@5": hits[5] / n if n else 0,
        "Hit@10": hits[10] / n if n else 0,
        "MRR": mrr_sum / n if n else 0,
        "nDCG@5": ndcg_sum / n if n else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=["dense", "hybrid", "bm25", "rerank"], default="dense")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    search = build_search(args.retriever)
    m = evaluate(search=search, limit=args.limit)
    print(f"retriever: {args.retriever} | questions: {m['n']}")
    for key in ("Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR", "nDCG@5"):
        print(f"  {key}: {m[key]:.4f}")


if __name__ == "__main__":
    main()
