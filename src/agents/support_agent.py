"""企业知识库智能客服 Agent（Agentic RAG + 意图路由）。

图结构：
    prepare → route ─(kb)→ retrieve → grade ─(相关)→ generate → END
                   │                      └(不相关 & 改写<2)→ rewrite → retrieve
                   │                      └(不相关 & 改写≥2)→ escalate → END
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

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph

from agents.tools import format_documents, search_knowledge_base_hybrid
from core import get_model, settings

MAX_REWRITES = 2  # 纠正式改写上限（业界常见 2-3 次，防死循环）

ROUTE_PROMPT = """你是客服意图分类器。判断用户这句话属于哪类：

- kb：与 Dify 产品功能/使用/部署/配置/插件相关，需要查知识库回答
- chitchat：寒暄、闲聊、问候、感谢（如"你好""谢谢""在吗"）
- escalate：投诉、辱骂，或超出客服范围（咨询其他产品、要求转人工等）

只输出一个词：kb / chitchat / escalate。"""

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

REWRITE_PROMPT = """原查询在知识库中检索效果不佳。请把它改写成一个更利于检索的查询（更具体、关键词更明确、去掉口语和无关词）。只输出改写后的查询，不要解释。"""

ESCALATE_TEXT = "抱歉，我在知识库中没能找到足以可靠回答这个问题的内容，已为您转接人工客服，请稍候。"

CHITCHAT_TEXT = "你好！我是 Dify 产品的智能客服，可以帮你解答 Dify 的使用、部署、配置、插件开发等问题。请问有什么可以帮你？"


class AgentState(MessagesState, total=False):
    """图状态：messages 来自 MessagesState（自动合并），其余为业务字段。"""

    question: str  # 当前检索 query（可能被 rewrite 改写）
    original_question: str  # 用户原始问题（route/generate 用它，不被改写污染）
    intent: str  # kb / chitchat / escalate
    documents: list[dict]
    relevant: bool
    rewrite_count: int


def _get_model(config: RunnableConfig) -> BaseChatModel:
    return get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))


def _last_human(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content)
    return ""


def prepare(state: AgentState, config: RunnableConfig) -> dict:
    """入口：取最新用户消息作为初始问题，重置改写计数。"""
    q = _last_human(state["messages"])
    return {"question": q, "original_question": q, "rewrite_count": 0}


async def route(state: AgentState, config: RunnableConfig) -> dict:
    """意图路由：kb（查知识库）/ chitchat（闲聊）/ escalate（越界，转人工）。

    解析失败默认走 kb，交给后面的 Corrective RAG 兜底（不相关会转人工）。
    """
    model = _get_model(config)
    response = await model.ainvoke(
        [SystemMessage(ROUTE_PROMPT), HumanMessage(state["original_question"])]
    )
    raw = str(response.content).strip().lower()
    if "chitchat" in raw:
        intent = "chitchat"
    elif "escalate" in raw:
        intent = "escalate"
    else:
        intent = "kb"
    return {"intent": intent}


def route_intent(state: AgentState) -> Literal["retrieve", "chitchat", "escalate"]:
    intent = state["intent"]
    if intent == "chitchat":
        return "chitchat"
    if intent == "escalate":
        return "escalate"
    return "retrieve"


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
    """生成节点：基于检索文档回答原始问题，标注来源。"""
    model = _get_model(config)
    prompt = GENERATE_PROMPT.format(
        question=state["original_question"],
        documents=format_documents(state["documents"]),
    )
    response = await model.ainvoke([SystemMessage(prompt)])
    return {"messages": [response]}


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


def route_after_grade(state: AgentState) -> Literal["generate", "rewrite", "escalate"]:
    if state["relevant"]:
        return "generate"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "escalate"


graph = StateGraph(AgentState)
graph.add_node("prepare", prepare)
graph.add_node("route", route)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)
graph.add_node("escalate", escalate)
graph.add_node("chitchat", chitchat)
graph.set_entry_point("prepare")
graph.add_edge("prepare", "route")
graph.add_conditional_edges(
    "route",
    route_intent,
    {"retrieve": "retrieve", "chitchat": "chitchat", "escalate": "escalate"},
)
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges(
    "grade",
    route_after_grade,
    {"generate": "generate", "rewrite": "rewrite", "escalate": "escalate"},
)
graph.add_edge("rewrite", "retrieve")
graph.add_edge("generate", END)
graph.add_edge("escalate", END)
graph.add_edge("chitchat", END)

support_agent = graph.compile()
