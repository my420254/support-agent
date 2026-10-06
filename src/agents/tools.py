"""检索工具：封装 Qdrant 检索 + 结果格式化。

P3 的 Agentic 图直接调用 search_knowledge_base（显式节点），
而不是让模型自己决定调不调工具 —— 这样检索步骤可独立评估、可观测。
P5 多 agent（订单/退款专家）会重新引入 function calling。
"""

from retrieval.embedder import get_embedder
from retrieval.qdrant_store import QdrantStore

_embedder = None
_store: QdrantStore | None = None


def _get_retrieval():
    global _embedder, _store
    if _embedder is None:
        _embedder = get_embedder()
    if _store is None:
        _store = QdrantStore()
    return _embedder, _store


def search_knowledge_base(query: str, top_k: int = 4) -> list[dict]:
    """检索知识库，返回 [{title, url, text, score}]。"""
    embedder, store = _get_retrieval()
    results = store.search(embedder.embed_query(query), limit=top_k)
    return [
        {
            "doc_id": r.payload["doc_id"],
            "title": r.payload["title"],
            "url": r.payload["url"],
            "text": r.payload["text"],
            "score": r.score,
        }
        for r in results
    ]


_hybrid = None


def _get_hybrid():
    """懒加载 HybridRetriever（首次扫全量 chunk 建 BM25，几秒）。"""
    global _hybrid
    if _hybrid is None:
        from retrieval.hybrid import HybridRetriever

        _hybrid = HybridRetriever()
    return _hybrid


def search_knowledge_base_hybrid(query: str, top_k: int = 4) -> list[dict]:
    """混合检索（dense + BM25 + RRF）。线上 agent 用它（eval 证明优于纯 dense）。"""
    return _get_hybrid().search(query, top_k=top_k)


def format_documents(docs: list[dict]) -> str:
    """把检索结果格式化为给模型看的上下文（带标题/来源/内容）。"""
    if not docs:
        return "（无检索结果）"
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[文档{i}] 标题: {d['title']}\n来源: {d['url']}\n内容: {d['text']}")
    return "\n\n".join(parts)


# --- 业务工具（mock 订单/工单系统；生产替换为真实 API）---

_MOCK_ORDERS = {
    "ORD12345": {"status": "已发货", "item": "Dify 企业版订阅", "eta": "2026-10-08"},
    "ORD67890": {"status": "待付款", "item": "Dify 专业版订阅", "eta": None},
}


def query_order(order_id: str) -> str:
    """查询订单状态（mock 订单系统）。生产替换为订单系统 API。"""
    o = _MOCK_ORDERS.get(order_id.strip().upper())
    if not o:
        return f"未找到订单 {order_id}，请核对订单号后重试。"
    eta = f"，预计送达 {o['eta']}" if o.get("eta") else ""
    return f"订单 {order_id}：商品「{o['item']}」，状态「{o['status']}」{eta}。"


def create_ticket(category: str, summary: str) -> str:
    """创建工单（mock 工单系统），返回工单号。生产替换为工单系统 API。"""
    from uuid import uuid4

    return "TICKET-" + uuid4().hex[:6].upper()


# --- 工具契约注册（单点执行器；详见 tool_runtime.py 的设计说明）---

from agents.tool_runtime import RiskClass, ToolSpec, ToolResult, executor  # noqa: E402

executor.register(ToolSpec(
    name="lookup_knowledge_base", risk=RiskClass.READ, fn=search_knowledge_base_hybrid,
    description="检索 Dify 知识库", timeout_s=15.0, max_attempts=2,
))
executor.register(ToolSpec(
    name="query_order", risk=RiskClass.READ, fn=query_order,
    description="查询订单状态", timeout_s=5.0, max_attempts=3,
))
executor.register(ToolSpec(
    name="create_ticket", risk=RiskClass.WRITE, fn=create_ticket,
    description="创建工单（写操作，需幂等）", timeout_s=5.0, max_attempts=2,
))


def call_tool(name: str, args: dict, *, thread_id: str = "", step: int = 0) -> ToolResult:
    """经单点执行器调用工具（校验/幂等/重试/审计全在这里）。"""
    return executor.execute(name, args, thread_id=thread_id, step=step)
