"""混合检索：dense(Qdrant) + sparse(BM25) → RRF 融合。

RRF（Reciprocal Rank Fusion）：两路检索各自按相关性排名，融合分 = Σ 1/(k+rank)，
再按融合分重排（k 通常取 60）。相比单路 dense，关键词类问题（部署/报错/专有名词）召回更准。

为什么需要 BM25：dense 向量擅长语义相近，但对精确关键词（命令、版本号、专有名词）
不敏感；BM25 恰好补这一块，二者 RRF 融合通常能同时提升 Recall 和 MRR。
"""

import jieba
from rank_bm25 import BM25Okapi

from retrieval.embedder import Embedder
from retrieval.qdrant_store import QdrantStore

RRF_K = 60  # RRF 常数，业界常用 60
DENSE_LIMIT = 20  # dense 路取的候选数
SPARSE_LIMIT = 20  # BM25 路取的候选数


class HybridRetriever:
    """dense + BM25 混合检索。首次使用从 Qdrant 拉全量 chunk 建 BM25 索引（几秒）。"""

    def __init__(self):
        self.embedder = Embedder()
        self.store = QdrantStore()
        self.chunks: list = []
        self.bm25: BM25Okapi | None = None
        self._build_bm25()

    def _build_bm25(self) -> None:
        """从 Qdrant 拉全量 chunk 文本（与 dense 同一份数据），jieba 分词后建 BM25。"""
        chunks = []
        offset = None
        while True:
            res, offset = self.store.client.scroll(
                collection_name=self.store.collection,
                limit=1000,
                offset=offset,
                with_payload=True,
            )
            chunks.extend(res)
            if offset is None or not res:
                break
        self.chunks = chunks
        corpus = [list(jieba.cut(c.payload["text"])) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    @staticmethod
    def _format(point) -> dict:
        p = point.payload
        return {
            "doc_id": p["doc_id"],
            "title": p["title"],
            "url": p["url"],
            "text": p["text"],
        }

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        # 1) dense 路：向量检索
        vec = self.embedder.embed_query(query)
        dense = self.store.search(vec, limit=DENSE_LIMIT)

        # 2) sparse 路：BM25 关键词检索
        bm25_scores = self.bm25.get_scores(list(jieba.cut(query)))
        sparse_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:SPARSE_LIMIT]

        # 3) RRF 融合（用 Qdrant point id 作为 chunk 唯一键）
        fused: dict = {}
        point_by_id: dict = {}
        for rank, pt in enumerate(dense):
            pid = pt.id
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
            point_by_id[pid] = pt
        for rank, idx in enumerate(sparse_idx):
            pt = self.chunks[idx]
            pid = pt.id
            fused[pid] = fused.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
            point_by_id[pid] = pt

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
        for pid, score in ordered:
            d = self._format(point_by_id[pid])
            d["score"] = round(score, 4)
            out.append(d)
        return out
