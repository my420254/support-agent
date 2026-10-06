"""客服 Agent 的工具集。

lookup_knowledge_base 做真实的 Qdrant 检索（当前为 dense 基线），
返回带来源 URL 的文档片段，供模型生成带引用的回答。
"""

from langchain_core.tools import tool

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


def search_knowledge_base(query: str, top_k: int = 4) -> list:
    embedder, store = _get_retrieval()
    return store.search(embedder.embed_query(query), limit=top_k)


@tool
def lookup_knowledge_base(query: str) -> str:
    """在 Dify 知识库中检索与 query 相关的文档。

    返回最相关的若干文档片段及其来源 URL。凡是产品功能、配置、部署、政策类问题，
    都应先调用本工具，再基于返回内容回答并标注来源。
    """
    results = search_knowledge_base(query)
    if not results:
        return "知识库中没有检索到相关内容，请如实告知用户并建议转人工。"

    blocks = []
    for i, r in enumerate(results, 1):
        p = r.payload
        blocks.append(f"[{i}] {p['title']}\n来源: {p['url']}\n{p['text']}")
    return "\n\n".join(blocks)
