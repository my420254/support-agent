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

GRADE_PROMPT = """你是检索质量评审。判断下面的检索文档是否足以回答用户问题。

用户问题：{question}

检索到的文档：
{documents}

只输出一个词：YES（文档足以给出有依据的回答）或 NO（不足以回答）。不要输出任何其他内容。"""

GENERATE_PROMPT = """你是「Dify」产品的技术支持客服。基于下面检索到的文档回答用户问题。

规则：
1. 只用文档里的信息回答，不要编造。
2. 引用来源时标注文档给出的 URL。
3. 文档不足以回答时，明确说明并建议转人工。

用户问题：{question}

检索到的文档：
{documents}"""

VERIFY_PROMPT = """你是答案审核员。判断下面的回答中的关键事实是否都能在文档上下文中找到依据（没有编造）。

注意：礼貌用语、格式、引导语不算编造；只关心事实性陈述是否有据可查。

文档上下文：
{documents}

待审回答：
{answer}

只输出一个词：YES（关键事实都有据可查）或 NO（回答包含编造/无依据的事实）。"""

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
    intent: str  # kb / chitchat / escalate
    documents: list[dict]
    relevant: bool
    rewrite_count: int
    draft_answer: str  # generate 的草稿，verify 通过后才入 messages
    verified: bool  # 自验证结果


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
    resp = await model.ainvoke(
        [SystemMessage(SAQ_PROMPT.format(history=history, last=str(messages[-1].content)))]
    )
    return str(resp.content).strip()


def prepare(state: AgentState, config: RunnableConfig) -> dict:
    """入口：取最新用户消息作为原始问题（original_question 保持字面，不被改写污染）。"""
    q = _last_human(state["messages"])
    return {"question": q, "original_question": q, "rewrite_count": 0}


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
    response = await model.ainvoke(
        [SystemMessage(ROUTE_PROMPT), HumanMessage(state["original_question"])]
    )
    raw = str(response.content).strip().lower()
    if "order" in raw:
        intent = "order"
    elif "complaint" in raw:
        intent = "complaint"
    elif "chitchat" in raw:
        intent = "chitchat"
    elif "escalate" in raw:
        intent = "escalate"
    else:
        intent = "kb"
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


async def grade(state: AgentState, config: RunnableConfig) -> dict:
    """评分节点：LLM 判断检索结果是否足以回答（YES/NO 纯文本 + 解析）。"""
    model = _get_model(config)
    prompt = GRADE_PROMPT.format(
        question=state["original_question"],
        documents=format_documents(state["documents"]),
    )
    response = await model.ainvoke([SystemMessage(prompt)])
    relevant = str(response.content).strip().lower().startswith("yes")
    return {"relevant": relevant}


async def generate(state: AgentState, config: RunnableConfig) -> dict:
    """生成节点：基于检索文档回答原始问题，先写成草稿（draft_answer），待 verify 通过。"""
    model = _get_model(config)
    prompt = GENERATE_PROMPT.format(
        question=state["original_question"],
        documents=format_documents(state["documents"]),
    )
    response = await model.ainvoke([SystemMessage(prompt)])
    return {"draft_answer": str(response.content)}


async def verify(state: AgentState, config: RunnableConfig) -> dict:
    """自验证（Self-Reflection）：检查草稿是否被检索文档支撑，防幻觉。"""
    model = _get_model(config)
    prompt = VERIFY_PROMPT.format(
        documents=format_documents(state["documents"]),
        answer=state["draft_answer"],
    )
    response = await model.ainvoke([SystemMessage(prompt)])
    return {"verified": str(response.content).strip().lower().startswith("yes")}


def commit(state: AgentState, config: RunnableConfig) -> dict:
    """验证通过：把草稿写入 messages 交付给用户。"""
    return {"messages": [AIMessage(content=state["draft_answer"])]}


async def rewrite(state: AgentState, config: RunnableConfig) -> dict:
    """纠正式改写节点：改写检索 query（不影响 original_question）后重检索。"""
    model = _get_model(config)
    response = await model.ainvoke(
        [SystemMessage(REWRITE_PROMPT), HumanMessage(state["question"])]
    )
    return {
        "question": str(response.content).strip(),
        "rewrite_count": state["rewrite_count"] + 1,
    }


def escalate(state: AgentState, config: RunnableConfig) -> dict:
    """转人工节点：诚实拒答。"""
    return {"messages": [AIMessage(content=ESCALATE_TEXT)]}


def chitchat(state: AgentState, config: RunnableConfig) -> dict:
    """闲聊节点：固定兜底回复，不走检索。"""
    return {"messages": [AIMessage(content=CHITCHAT_TEXT)]}


def order_lookup(state: AgentState, config: RunnableConfig) -> dict:
    """订单查询节点：实体抽取（订单号）→ 调订单工具 → 返回结果。"""
    m = ORDER_ID_RE.search(state["original_question"])
    if not m:
        return {
            "messages": [
                AIMessage(content="请提供订单号（格式如 ORD12345），我来帮您查询订单状态。")
            ]
        }
    return {"messages": [AIMessage(content=query_order(m.group(0)))]}


def complaint_handle(state: AgentState, config: RunnableConfig) -> dict:
    """投诉处理节点：创建工单（工具调用）+ 转人工。"""
    ticket_id = create_ticket("投诉", state["original_question"][:80])
    return {
        "messages": [
            AIMessage(
                content=f"非常抱歉给您带来不便。已为您创建工单 {ticket_id} 并转接人工客服优先处理，请稍候。"
            )
        ]
    }


def route_after_grade(state: AgentState) -> Literal["generate", "rewrite", "escalate"]:
    if state["relevant"]:
        return "generate"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "escalate"


def route_after_verify(state: AgentState) -> Literal["commit", "escalate"]:
    return "commit" if state["verified"] else "escalate"


def _build_knowledge_specialist():
    """知识专家子图：SAQ → Corrective RAG → 自验证。独立编译，可单独替换/测试。"""
    g = StateGraph(AgentState)
    g.add_node("resolve_query", resolve_query)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("generate", generate)
    g.add_node("verify", verify)
    g.add_node("commit", commit)
    g.add_node("rewrite", rewrite)
    g.add_node("escalate", escalate)
    g.set_entry_point("resolve_query")
    g.add_edge("resolve_query", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", route_after_grade, {"generate": "generate", "rewrite": "rewrite", "escalate": "escalate"}
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", route_after_verify, {"commit": "commit", "escalate": "escalate"})
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
graph.add_node("route", route)
graph.add_node("knowledge", _build_knowledge_specialist())
graph.add_node("order", _build_tool_specialist("order_lookup", order_lookup))
graph.add_node("complaint", _build_tool_specialist("complaint_handle", complaint_handle))
graph.add_node("chitchat", chitchat)
graph.add_node("escalate", escalate)
graph.set_entry_point("prepare")
graph.add_edge("prepare", "route")
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
