"""切块策略消融：分块方式 / chunk_size / overlap 对检索指标的影响。

之前只用了「标题感知 400/60」一种配置，没有做消融——这个脚本补上。
做法：对每组配置，用临时 collection 重建索引，再用 golden 120 题跑检索评测。

用法：python -m eval.chunk_ablation [--configs 5]
"""

import json
import sys
import time
import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from eval.metrics import hit_at_k, ndcg_at_k, reciprocal_rank, unique_doc_ids
from ingestion.chunker import chunk_markdown
from retrieval.embedder import Embedder
from retrieval.qdrant_store import QdrantStore

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
GOLDEN = ROOT / "data" / "golden" / "golden.jsonl"

# (策略, chunk_size, overlap)
CONFIGS = [
    ("heading", 400, 60),  # 当前线上配置（基线）
    ("fixed", 400, 60),    # 朴素定长（对照组）
    ("heading", 200, 40),  # 更小 chunk
    ("heading", 800, 100), # 更大 chunk
    ("heading", 400, 0),   # 无重叠
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


def build_temp_index(corpus: list[dict], embedder: Embedder, strategy: str, size: int, overlap: int):
    """按指定切块策略建临时 collection，返回 (QdrantStore, 集合名, chunk 数)。"""
    chunks = []
    for doc in corpus:
        for c in chunk_markdown(doc["content"], chunk_size=size, overlap=overlap, strategy=strategy):
            embed_text = (c["heading"] + "\n" + c["text"]) if c["heading"] else c["text"]
            chunks.append({**doc, "heading": c["heading"], "text": c["text"], "embed_text": embed_text})

    collection = f"ablate_{strategy}_{size}_{overlap}_{uuid.uuid4().hex[:6]}"
    store = QdrantStore(collection=collection)
    store.recreate(embedder.dim)
    vecs = embedder.embed_documents([c["embed_text"] for c in chunks])
    store.upsert([
        PointStruct(
            id=str(uuid.uuid4()),
            vector=v,
            payload={"doc_id": c["id"], "title": c["title"], "url": c["url"],
                     "heading": c["heading"], "text": c["text"]},
        )
        for c, v in zip(chunks, vecs)
    ])
    return store, collection, len(chunks)


def eval_dense(store: QdrantStore, embedder: Embedder, golden: list[dict], k: int = 10) -> dict:
    n = len(golden)
    h1 = h3 = h5 = 0
    mrr = ndcg = 0.0
    for g in golden:
        res = store.search(embedder.embed_query(g["question"]), limit=k)
        docs = unique_doc_ids([{"doc_id": p.payload["doc_id"]} for p in res])
        h1 += hit_at_k(g["gt_doc_id"], docs, 1)
        h3 += hit_at_k(g["gt_doc_id"], docs, 3)
        h5 += hit_at_k(g["gt_doc_id"], docs, 5)
        mrr += reciprocal_rank(g["gt_doc_id"], docs)
        ndcg += ndcg_at_k(g["gt_doc_id"], docs, 5)
    return {"Hit@1": h1 / n, "Hit@3": h3 / n, "Hit@5": h5 / n, "MRR": mrr / n, "nDCG@5": ndcg / n}


def main() -> None:
    corpus = load_jsonl(CORPUS)
    golden = load_jsonl(GOLDEN)
    embedder = Embedder()
    print(f"语料 {len(corpus)} 篇 | 评测 {len(golden)} 题 | 共 {len(CONFIGS)} 组配置\n")

    rows = []
    for strategy, size, overlap in CONFIGS:
        t0 = time.time()
        store, collection, nchunks = build_temp_index(corpus, embedder, strategy, size, overlap)
        m = eval_dense(store, embedder, golden)
        rows.append((strategy, size, overlap, nchunks, m))
        print(f"{strategy:8s} size={size:4d} overlap={overlap:3d} | {nchunks:5d} chunks | "
              f"Hit@1={m['Hit@1']:.4f} Hit@3={m['Hit@3']:.4f} Hit@5={m['Hit@5']:.4f} "
              f"MRR={m['MRR']:.4f} nDCG@5={m['nDCG@5']:.4f} | {time.time()-t0:.0f}s")
        store.client.delete_collection(collection)  # 清理临时集合

    best = max(rows, key=lambda r: r[4]["MRR"])
    print(f"\n最优（按 MRR）: {best[0]} size={best[1]} overlap={best[2]} → MRR={best[4]['MRR']:.4f}")


if __name__ == "__main__":
    main()
