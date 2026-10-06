"""RRF 权重调参实验：等权 RRF 是否已是最优？weighted RRF 能否提升？

方法（避免调参过拟合）：
1. 120 题 golden 集切分：80 题验证集 + 40 题测试集（固定 seed）；
2. 在验证集上网格搜索 (alpha, k)：
     fused = alpha/(k + rank_dense) + (1-alpha)/(k + rank_sparse)
   alpha=0.5 即等权 RRF（当前线上配置）；
3. 用验证集选出的最优参数，在**从未参与调参的 40 题测试集**上报告最终结果。

用法：python -m eval.tune_weights
"""

import json
import random
import sys
from pathlib import Path

import jieba

from eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank
from retrieval.hybrid import RRF_K, HybridRetriever

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "golden" / "golden.jsonl"

DEPTH = 20  # 每路候选深度（与线上 DENSE_LIMIT/SPARSE_LIMIT 一致）


def load_golden() -> list[dict]:
    return [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l]


def precompute(hr: HybridRetriever, golden: list[dict]) -> list[dict]:
    """对每题只跑一次检索，缓存 dense/sparse 两路排名，供后续任意 alpha/k 复用。"""
    cache = []
    for g in golden:
        vec = hr.embedder.embed_query(g["question"])
        dense = hr.store.search(vec, limit=DEPTH)
        dense_ids = [pt.id for pt in dense]

        scores = hr.bm25.get_scores(list(jieba.cut(g["question"])))
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:DEPTH]
        sparse_ids = [hr.chunks[i].id for i in idx]

        id2doc = {pt.id: pt.payload["doc_id"] for pt in dense}
        for i in idx:
            id2doc[hr.chunks[i].id] = hr.chunks[i].payload["doc_id"]

        cache.append(
            {"gt": g["gt_doc_id"], "dense": dense_ids, "sparse": sparse_ids, "id2doc": id2doc}
        )
    return cache


def fuse_docs(item: dict, alpha: float, k: int) -> list[str]:
    """按 alpha/k 融合两路排名 → 取 top10 chunk → 去重成文档级有序列表。"""
    fused: dict = {}
    for rank, pid in enumerate(item["dense"]):
        fused[pid] = fused.get(pid, 0.0) + alpha / (k + rank + 1)
    for rank, pid in enumerate(item["sparse"]):
        fused[pid] = fused.get(pid, 0.0) + (1 - alpha) / (k + rank + 1)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:10]
    seen, docs = set(), []
    for pid, _ in ordered:
        d = item["id2doc"].get(pid)
        if d and d not in seen:
            seen.add(d)
            docs.append(d)
    return docs


def score(cache: list[dict], alpha: float, k: int) -> dict:
    n = len(cache)
    h1 = h3 = h5 = 0
    mrr = ndcg = 0.0
    for item in cache:
        docs = fuse_docs(item, alpha, k)
        h1 += hit_at_k(item["gt"], docs, 1)
        h3 += hit_at_k(item["gt"], docs, 3)
        h5 += hit_at_k(item["gt"], docs, 5)
        mrr += reciprocal_rank(item["gt"], docs)
        ndcg += ndcg_at_k(item["gt"], docs, 5)
    return {
        "Hit@1": h1 / n, "Hit@3": h3 / n, "Hit@5": h5 / n,
        "MRR": mrr / n, "nDCG@5": ndcg / n,
    }


def main() -> None:
    golden = load_golden()
    rng = random.Random(42)
    rng.shuffle(golden)
    val, test = golden[:80], golden[80:]  # 80 验证 / 40 测试

    hr = HybridRetriever()
    print("预计算两路排名（每题只检索一次）...")
    val_cache = precompute(hr, val)
    test_cache = precompute(hr, test)

    # 网格搜索：alpha（dense 权重）× k（RRF 平滑常数）
    alphas = [round(i / 10, 1) for i in range(11)]
    ks = [10, 30, 60, 100]
    print("\n=== 验证集(80题) 网格搜索 ===")
    results = []
    for a in alphas:
        for k in ks:
            m = score(val_cache, a, k)
            results.append((a, k, m))
    results.sort(key=lambda x: (x[2]["MRR"], x[2]["Hit@1"]), reverse=True)
    for a, k, m in results[:5]:
        tag = "  <- 等权基准(alpha=0.5,k=60)" if (a == 0.5 and k == 60) else ""
        print(f"  alpha={a} k={k}: Hit@1={m['Hit@1']:.4f} MRR={m['MRR']:.4f} nDCG@5={m['nDCG@5']:.4f}{tag}")

    best_a, best_k, _ = results[0]
    base = score(val_cache, 0.5, RRF_K)
    print(f"\n验证集最优: alpha={best_a}, k={best_k}")
    print(f"  等权基准: Hit@1={base['Hit@1']:.4f} MRR={base['MRR']:.4f}")
    print(f"  调参最优: Hit@1={results[0][2]['Hit@1']:.4f} MRR={results[0][2]['MRR']:.4f}")

    # 测试集（从未参与调参）最终报告
    print("\n=== 测试集(40题，held-out) ===")
    for label, a, k in [
        ("等权基准(0.5, 60)", 0.5, RRF_K),
        ("调参最优(0.1, 100)", best_a, best_k),
        ("保守选择(0.1, 60)", 0.1, RRF_K),
    ]:
        t = score(test_cache, a, k)
        print(f"  {label}: Hit@1={t['Hit@1']:.4f} Hit@3={t['Hit@3']:.4f} Hit@5={t['Hit@5']:.4f} MRR={t['MRR']:.4f} nDCG@5={t['nDCG@5']:.4f}")


if __name__ == "__main__":
    main()
