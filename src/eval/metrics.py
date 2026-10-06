"""检索评测指标（文档级）。

golden set 每条问题有一个 ground-truth 文档 id（gt_doc_id）。
文档级：把 top-k 检索结果里出现过的唯一 doc_id 当作"命中集合"，
比 chunk 级更稳健（同一文档可能有多个相关 chunk）。
"""


def unique_doc_ids(results: list[dict]) -> list[str]:
    """按顺序去重，保留文档首次出现的位置。"""
    seen: list[str] = []
    for r in results:
        if r["doc_id"] not in seen:
            seen.append(r["doc_id"])
    return seen


def hit_at_k(gt_doc_id: str, retrieved_doc_ids: list[str], k: int) -> int:
    return int(gt_doc_id in retrieved_doc_ids[:k])


def reciprocal_rank(gt_doc_id: str, retrieved_doc_ids: list[str]) -> float:
    for i, d in enumerate(retrieved_doc_ids, 1):
        if d == gt_doc_id:
            return 1.0 / i
    return 0.0


def ndcg_at_k(gt_doc_id: str, retrieved_doc_ids: list[str], k: int) -> float:
    """单 gt 的二值相关 nDCG@k：命中第 r 位得 1/log2(r+1)，未命中 0。"""
    import math

    for i, d in enumerate(retrieved_doc_ids[:k], 1):
        if d == gt_doc_id:
            return 1.0 / math.log2(i + 1)
    return 0.0
