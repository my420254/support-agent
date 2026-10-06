"""混合检索：dense(Qdrant) + sparse(BM25) → weighted RRF 融合。

RRF（Reciprocal Rank Fusion）：两路各自按名次贡献 1/(k+rank)，加权后融合再重排。
用「名次」而非「分值」，回避了 BM25 分值(0~几十)与 cosine(0~1) 量纲不同、无法直接加权的问题。

权重来源（eval/tune_weights.py，80 验证/40 测试切分）：
  调参得 dense 权重 alpha=0.1、BM25 权重 0.9 —— 与消融实验(BM25>dense)相互印证，
  测试集 Hit@1 0.750→0.775、nDCG@5 0.889→0.911。k 在 60~100 区间不敏感。
"""

import re

import jieba
from rank_bm25 import BM25Okapi

from retrieval.embedder import get_embedder
from retrieval.qdrant_store import QdrantStore

# 中文段（含标点）用于分流；其余按英文/数字处理
_CJK_RE = re.compile(r"[一-鿿]+")
_TOKEN_RE = re.compile(r"[一-鿿]+|[A-Za-z0-9_]+")


def tokenize_mixed(text: str) -> list[str]:
    """中英混合分词：中文段走 jieba，英文/数字段按小写单词切。

    为什么不能只用 jieba：jieba 是中文分词器，对 "How to deploy Dify" 这类
    英文会把连续字母当成一个"词"，导致 BM25 无法命中单个英文关键词。
    """
    tokens: list[str] = []
    for seg in _TOKEN_RE.findall(text):
        if _CJK_RE.match(seg):
            tokens.extend(jieba.cut(seg))
        else:
            tokens.append(seg.lower())
    return tokens

RRF_K = 60  # RRF 平滑常数（原论文/ES/Qdrant 默认；60~100 不敏感）
DENSE_LIMIT = 20  # dense 路取的候选数
SPARSE_LIMIT = 20  # BM25 路取的候选数
DENSE_WEIGHT = 0.1  # dense 权重 alpha；BM25 权重 = 1 - alpha = 0.9（调参得出）


class HybridRetriever:
    """dense + BM25 混合检索。首次使用从 Qdrant 拉全量 chunk 建 BM25 索引（几秒）。"""

    def __init__(self):
        self.embedder = get_embedder()
        self.store = QdrantStore()
        self.chunks: list = []
        self.bm25: BM25Okapi | None = None
        self._reranker = None
        self._build_bm25()

    def _get_reranker(self):
        if self._reranker is None:
            from retrieval.reranker import Reranker

            self._reranker = Reranker()
        return self._reranker

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
        corpus = [tokenize_mixed(c.payload["text"]) for c in chunks]
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

    def search_bm25(self, query: str, top_k: int = 4) -> list[dict]:
        """仅 BM25 关键词检索（用于消融实验，量化 dense 与 BM25 各自贡献）。"""
        scores = self.bm25.get_scores(tokenize_mixed(query))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for idx in top_idx:
            d = self._format(self.chunks[idx])
            d["score"] = round(float(scores[idx]), 4)
            out.append(d)
        return out

    def search_rerank(self, query: str, top_k: int = 4, candidate_k: int = 20) -> list[dict]:
        """hybrid 召回 candidate_k 条 → cross-encoder 重排 → 取 top_k。"""
        candidates = self.search(query, top_k=candidate_k)
        return self._get_reranker().rerank(query, candidates, top_k)

    def search(self, query: str, top_k: int = 4, alpha: float = DENSE_WEIGHT) -> list[dict]:
        # 1) dense 路：向量检索
        vec = self.embedder.embed_query(query)
        dense = self.store.search(vec, limit=DENSE_LIMIT)

        # 2) sparse 路：BM25 关键词检索
        bm25_scores = self.bm25.get_scores(tokenize_mixed(query))
        sparse_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:SPARSE_LIMIT]

        # 3) weighted RRF 融合（alpha=dense 权重，1-alpha=BM25 权重）
        fused: dict = {}
        point_by_id: dict = {}
        for rank, pt in enumerate(dense):
            pid = pt.id
            fused[pid] = fused.get(pid, 0.0) + alpha / (RRF_K + rank + 1)
            point_by_id[pid] = pt
        for rank, idx in enumerate(sparse_idx):
            pt = self.chunks[idx]
            pid = pt.id
            fused[pid] = fused.get(pid, 0.0) + (1 - alpha) / (RRF_K + rank + 1)
            point_by_id[pid] = pt

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
        for pid, score in ordered:
            d = self._format(point_by_id[pid])
            d["score"] = round(score, 4)
            out.append(d)
        return out
