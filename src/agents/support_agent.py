"""企业知识库智能客服 Agent（Agentic RAG + 意图路由）。

图结构：
    prepare → route ─(kb)→ resolve_query → retrieve → grade ─(相关)→ generate → END
                   │                                        └(不相关&<2)→ rewrite → retrieve
                   │                                        └(不相关&≥2)→ escalate → END
                   ├(chitchat)→ chitchat → END
                   └(escalate)→ escalate → END

设计思想：
1. route 做「输入理解」——意图分类（kb 知识库问题 / chitchat 闲聊 / escalate 越界投诉），
   把「路由」和「生成」分开（业界原则：orchestration 决策独立于措辞）。
   闲聊/越界直接短路，不再浪费检索。
2. Corrective RAG：检索不相关就改写 query 重检索（限 MAX_REWRITES 次），仍不相关转人工。
3. 检索用 hybrid（dense + BM25 + RRF）；回答用 original_question（不被改写污染）。

踩坑记录（DeepSeek v4.1-flash 是推理模型）：不支持 response_format 与强制 tool_choice，
所以意图/评分都用「纯文本 + 解析」，不用 structured output。
"""

import re
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt

from agents.guardrails import detect_prompt_injection, mask_pii
from agents.tools import create_ticket, format_documents, query_order, search_knowledge_base_hybrid
from core import get_model, settings

MAX_REWRITES = 2  # 纠正式改写上限（业界常见 2-3 次，防死循环）

ROUTE_PROMPT = """你是客服意图分类器。判断用户这句话属于哪类：

- kb：与 Dify 产品功能/使用/部署/配置/插件相关，需要查知识库回答
- order：查询订单状态、物流、退换货进度（通常带订单号）
- complaint：投诉、退款、赔偿等诉求
- chitchat：寒暄、闲聊、问候、感谢（如"你好""谢谢""在吗"）
- escalate：辱骂，或完全超出客服范围（咨询其他产品等）

只输出一个词：kb / order / complaint / chitchat / escalate。"""

ORDER_ID_RE = re.compile(r"ORD\d+", re.IGNORECASE)

EXTRACT_ORDER_ID_PROMPT = """从用户消息里提取订单号（形如 ORD12345）。只输出订单号本身；没有则输出 NONE。"""

GENERATE_PROMPT = """你是「Dify」产品的技术支持客服。基于下面检索到的文档回答用户问题。

规则：
1. 只用文档里的信息回答，不要编造。
2. 引用来源时标注文档给出的 URL。
3. 如果文档不足以回答，只输出「无法回答」四个字，不要猜测或补充。

用户问题：{question}

检索到的文档：
{documents}"""

REWRITE_PROMPT = """原查询在知识库中检索效果不佳。请把它改写成一个更利于检索的查询（更具体、关键词更明确、去掉口语和无关词）。只输出改写后的查询，不要解释。"""

SAQ_PROMPT = """把用户最新这句话改写成一句完整、独立、可检索的问句（结合对话历史补全省略的主语和指代）。

对话历史：
{history}

用户最新消息：{last}

只输出改写后的问句。"""

ESCALATE_TEXT = "抱歉，这个问题需要人工客服为您进一步处理，已为您转接人工客服，请稍候。"

CHITCHAT_TEXT = "你好！我是 Dify 产品的智能客服，可以帮你解答 Dify 的使用、部署、配置、插件开发等问题。请问有什么可以帮你？"


class AgentState(MessagesState, total=False):
    """图状态：messages 来自 MessagesState（自动合并），其余为业务字段。"""

    question: str  # 当前检索 query（可能被 rewrite 改写）
    original_question: str  # 用户原始问题（route/generate 用它，不被改写污染）
    intent: str  # kb / order / complaint / chitchat / escalate
    documents: list[dict]
    rewrite_count: int
    draft_answer: str  # generate 草稿，非「无法回答」时 commit 入 messages
    order_id: str  # 订单号（实体抽取结果）


def _get_model(config: RunnableConfig) -> BaseChatModel:
    return get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))


def _last_human(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def _has_prior_turn(messages: list) -> bool:
    """是否超过一轮对话（需要做指代消解）。"""
    return sum(1 for m in messages if isinstance(m, HumanMessage)) > 1


async def _standalone_query(messages: list, config: RunnableConfig) -> str:
    """SAQ：结合对话历史把最新消息改写成独立 query（补全省略/指代）。"""
    model = _get_model(config)
    history = "\n".join(f"{m.type}: {str(m.content)[:200]}" for m in messages[:-1])
    resp = await model.with_config(tags=["skip_stream"]).ainvoke(
        [SystemMessage(SAQ_PROMPT.format(history=history, last=str(messages[-1].content)))]
    )
    return str(resp.content).strip()


def prepare(state: AgentState, config: RunnableConfig) -> dict:
    """入口：取最新用户消息，PII 脱敏，作为原始问题（不被改写污染）。"""
    q = mask_pii(_last_human(state["messages"]))
    return {"question": q, "original_question": q, "rewrite_count": 0}


def guard(state: AgentState, config: RunnableConfig) -> dict:
    """安全护栏：Prompt Injection 检测 → 直接转人工，不进检索/工具。"""
    if detect_prompt_injection(state["original_question"]):
        return {"intent": "escalate"}
    return {}


def route_after_guard(state: AgentState) -> Literal["escalate", "route"]:
    return "escalate" if state.get("intent") == "escalate" else "route"


async def resolve_query(state: AgentState, config: RunnableConfig) -> dict:
    """多轮时把 question 改写成独立 query（SAQ），仅用于检索；不动 original_question。

    放在 route 之后：只有意图为 kb 才走这里，闲聊/投诉在 route 已短路，不会被 SAQ 误改。
    """
    if _has_prior_turn(state["messages"]):
        return {"question": await _standalone_query(state["messages"], config)}
    return {}


async def route(state: AgentState, config: RunnableConfig) -> dict:
    """意图路由：kb（查知识库）/ chitchat（闲聊）/ escalate（越界，转人工）。

    解析失败默认走 kb，交给后面的 Corrective RAG 兜底（不相关会转人工）。
    """
    model = _get_model(config)
    response = await model.with_config(tags=["skip_stream"]).ainvoke(
        [SystemMessage(ROUTE_PROMPT), HumanMessage(state["original_question"])]
    )
    raw = str(response.content).strip().lower()
    # 剥离可能的思考标签（如 deepseek reasoner），提取标准意图词
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    matches = re.findall(r"\b(kb|order|complaint|chitchat|escalate)\b", cleaned)
    intent = matches[0] if matches else "kb"
    return {"intent": intent}


def route_intent(state: AgentState) -> Literal["knowledge", "order", "complaint", "chitchat", "escalate"]:
    intent = state["intent"]
    if intent == "order":
        return "order"
    if intent == "complaint":
        return "complaint"
    if intent == "chitchat":
        return "chitchat"
    if intent == "escalate":
        return "escalate"
    return "knowledge"


async def retrieve(state: AgentState, config: RunnableConfig) -> dict:
    """检索节点：对当前问题做 hybrid 检索（dense + BM25 + RRF）。"""
    return {"documents": search_knowledge_base_hybrid(state["question"], top_k=4)}


async def generate(state: AgentState, config: RunnableConfig) -> dict:
    """生成节点：prompt 强制"只用文档、不足则输出「无法回答」"，一步同时完成评分+接地。"""
    model = _get_model(config)
    prompt = GENERATE_PROMPT.format(
        question=state["original_question"],
        documents=format_documents(state["documents"]),
    )
    response = await model.with_config(tags=["skip_stream"]).ainvoke([SystemMessage(prompt)])
    return {"draft_answer": str(response.content)}


def commit(state: AgentState, config: RunnableConfig) -> dict:
    """把草稿写入 messages 交付给用户。"""
    return {"messages": [AIMessage(content=state["draft_answer"])]}


async def rewrite(state: AgentState, config: RunnableConfig) -> dict:
    """纠正式改写节点：改写检索 query（不影响 original_question）后重检索。"""
    model = _get_model(config)
    response = await model.with_config(tags=["skip_stream"]).ainvoke(
        [SystemMessage(REWRITE_PROMPT), HumanMessage(state["question"])]
    )
    return {
        "question": str(response.content).strip(),
        "rewrite_count": state["rewrite_count"] + 1,
    }


def escalate(state: AgentState, config: RunnableConfig) -> dict:
    """真实转人工（HITL）：interrupt() 挂起图，人工客服通过 /resume 恢复。

    服务层检测到中断后返回提示给用户；人工坐席用同 thread_id 调 /invoke 时，
    服务端用 Command(resume=...) 唤醒这里，human_reply 拿到人工回复。
    """
    ticket_id = create_ticket("人工支持", state["original_question"][:80])
    human_reply = interrupt(f"该问题已转接人工客服（工单 {ticket_id}），请稍候。")
    return {"messages": [AIMessage(content=f"[人工客服回复] {human_reply}")]}


def chitchat(state: AgentState, config: RunnableConfig) -> dict:
    """闲聊节点：固定兜底回复，不走检索。"""
    return {"messages": [AIMessage(content=CHITCHAT_TEXT)]}


async def extract_order_id(state: AgentState, config: RunnableConfig) -> dict:
    """实体抽取：先正则快速路径，再 LLM 兜底（处理"我的单号是 12345"这类变体）。"""
    m = ORDER_ID_RE.search(state["original_question"])
    if m:
        return {"order_id": m.group(0).upper()}
    model = _get_model(config)
    resp = await model.with_config(tags=["skip_stream"]).ainvoke(
        [SystemMessage(EXTRACT_ORDER_ID_PROMPT), HumanMessage(state["original_question"])]
    )
    oid = str(resp.content).strip()
    return {"order_id": oid if oid and "none" not in oid.lower() else ""}


def route_order(state: AgentState) -> Literal["query_order", "ask_id"]:
    return "query_order" if state.get("order_id") else "ask_id"


def query_order_node(state: AgentState, config: RunnableConfig) -> dict:
    return {"messages": [AIMessage(content=query_order(state["order_id"]))]}


def ask_id(state: AgentState, config: RunnableConfig) -> dict:
    return {"messages": [AIMessage(content="请提供订单号（格式如 ORD12345），我来帮您查询订单状态。")]}


def _build_order_specialist():
    """订单专家子图：实体抽取 → 有号查询 / 无号反问（感知→决策→行动）。"""
    g = StateGraph(AgentState)
    g.add_node("extract_id", extract_order_id)
    g.add_node("query_order", query_order_node)
    g.add_node("ask_id", ask_id)
    g.set_entry_point("extract_id")
    g.add_conditional_edges("extract_id", route_order, {"query_order": "query_order", "ask_id": "ask_id"})
    g.add_edge("query_order", END)
    g.add_edge("ask_id", END)
    return g.compile()


def complaint_handle(state: AgentState, config: RunnableConfig) -> dict:
    """投诉处理：创建工单（工具）+ interrupt 转人工（HITL）。"""
    ticket_id = create_ticket("投诉", state["original_question"][:80])
    human_reply = interrupt(f"非常抱歉给您带来不便。已创建工单 {ticket_id} 并转接人工客服，请稍候。")
    return {"messages": [AIMessage(content=f"[人工客服回复] {human_reply}")]}


def route_after_generate(state: AgentState) -> Literal["commit", "rewrite", "escalate"]:
    """生成后路由：正常 → commit；「无法回答」→ 纠正式改写或转人工（防死循环）。"""
    if "无法回答" not in state["draft_answer"]:
        return "commit"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "escalate"


def _build_knowledge_specialist():
    """知识专家子图：SAQ → 检索 → 生成（prompt 级接地；「无法回答」→ 纠错/转人工）。"""
    g = StateGraph(AgentState)
    g.add_node("resolve_query", resolve_query)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("commit", commit)
    g.add_node("rewrite", rewrite)
    g.add_node("escalate", escalate)
    g.set_entry_point("resolve_query")
    g.add_edge("resolve_query", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges(
        "generate", route_after_generate, {"commit": "commit", "rewrite": "rewrite", "escalate": "escalate"}
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("commit", END)
    g.add_edge("escalate", END)
    return g.compile()


def _build_tool_specialist(node_name: str, node_fn):
    """单节点工具专家子图（订单/工单）。"""
    g = StateGraph(AgentState)
    g.add_node(node_name, node_fn)
    g.set_entry_point(node_name)
    g.add_edge(node_name, END)
    return g.compile()


# Supervisor：意图路由 → 分发到各专家子图（多 Agent 协作）
graph = StateGraph(AgentState)
graph.add_node("prepare", prepare)
graph.add_node("guard", guard)
graph.add_node("route", route)
graph.add_node("knowledge", _build_knowledge_specialist())
graph.add_node("order", _build_order_specialist())
graph.add_node("complaint", _build_tool_specialist("complaint_handle", complaint_handle))
graph.add_node("chitchat", chitchat)
graph.add_node("escalate", escalate)
graph.set_entry_point("prepare")
graph.add_edge("prepare", "guard")
graph.add_conditional_edges("guard", route_after_guard, {"escalate": "escalate", "route": "route"})
graph.add_conditional_edges(
    "route",
    route_intent,
    {
        "knowledge": "knowledge",
        "order": "order",
        "complaint": "complaint",
        "chitchat": "chitchat",
        "escalate": "escalate",
    },
)
graph.add_edge("knowledge", END)
graph.add_edge("order", END)
graph.add_edge("complaint", END)
graph.add_edge("chitchat", END)
graph.add_edge("escalate", END)

support_agent = graph.compile()
