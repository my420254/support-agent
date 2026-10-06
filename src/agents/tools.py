"""检索工具：封装 Qdrant 检索 + 结果格式化。

P3 的 Agentic 图直接调用 search_knowledge_base（显式节点），
而不是让模型自己决定调不调工具 —— 这样检索步骤可独立评估、可观测。
P5 多 agent（订单/退款专家）会重新引入 function calling。
"""

from retrieval.embedder import Embedder
from retrieval.qdrant_store import QdrantStore

_embedder: Embedder | None = None
_store: QdrantStore | None = None


def _get_retrieval() -> tuple[Embedder, QdrantStore]:
    global _embedder, _store
    if _embedder is None:
        _embedder = Embedder()
    if _store is None:
        _store = QdrantStore()
    return _embedder, _store


def search_knowledge_base(query: str, top_k: int = 4) -> list[dict]:
    """检索知识库，返回 [{title, url, text, score}]。"""
    embedder, store = _get_retrieval()
    results = store.search(embedder.embed_query(query), limit=top_k)
    return [
        {
            "title": r.payload["title"],
            "url": r.payload["url"],
            "text": r.payload["text"],
            "score": r.score,
        }
        for r in results
    ]


def format_documents(docs: list[dict]) -> str:
    """把检索结果格式化为给模型看的上下文（带标题/来源/内容）。"""
    if not docs:
        return "（无检索结果）"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[文档{i}] 标题: {d['title']}\n来源: {d['url']}\n内容: {d['text']}")
    return "\n\n".join(parts)
