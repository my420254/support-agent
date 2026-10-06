"""企业知识库智能客服 Agent（P3：Agentic RAG / Corrective RAG）。

图结构：
    prepare → retrieve → grade ─(相关)→ generate → END
                           └(不相关 & 改写<MAX)→ rewrite → retrieve
                           └(不相关 & 改写≥MAX)→ escalate → END

为什么从 P2 的「模型↔工具」循环升级成显式图：
1. 检索/评分/生成拆成独立节点 → 分步评估（检索指标 vs 生成指标分开看）、分步观测；
2. Corrective（纠正式）：检索结果不相关就改写 query 重检索，最多 MAX_REWRITES 次，防死循环；
3. 仍不相关 → 转人工（诚实拒答）而不是编造 —— 这是客服 agent 的核心价值。

踩坑记录（DeepSeek v4.1-flash 是推理模型）：
- 不支持 response_format（JSON 模式）→ with_structured_output 默认会报 400；
- 不支持强制 tool_choice → with_structured_output(method="function_calling") 会报 400；
- 因此 grade 用「YES/NO 纯文本 + 解析」，不依赖 structured output。
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph

from agents.tools import format_documents, search_knowledge_base
from core import get_model, settings

MAX_REWRITES = 2  # 纠正式改写上限（业界常见 2-3 次，防死循环）

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


class AgentState(MessagesState, total=False):
    """图状态：messages 来自 MessagesState（自动合并），其余为业务字段。"""

    question: str
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
    return {"question": _last_human(state["messages"]), "rewrite_count": 0}


async def retrieve(state: AgentState, config: RunnableConfig) -> dict:
    """检索节点：对当前问题做 Qdrant 检索。"""
    return {"documents": search_knowledge_base(state["question"], top_k=4)}


async def grade(state: AgentState, config: RunnableConfig) -> dict:
    """评分节点：LLM 判断检索结果是否足以回答（YES/NO 纯文本 + 解析）。"""
    model = _get_model(config)
    prompt = GRADE_PROMPT.format(
        question=state["question"],
        documents=format_documents(state["documents"]),
    )
    response = await model.ainvoke([SystemMessage(prompt)])
    relevant = "yes" in str(response.content).strip().lower()
    return {"relevant": relevant}


async def generate(state: AgentState, config: RunnableConfig) -> dict:
    """生成节点：基于检索文档回答，标注来源。"""
    model = _get_model(config)
    prompt = GENERATE_PROMPT.format(
        question=state["question"],
        documents=format_documents(state["documents"]),
    )
    response = await model.ainvoke([SystemMessage(prompt)])
    return {"messages": [response]}


async def rewrite(state: AgentState, config: RunnableConfig) -> dict:
    """纠正式改写节点：改写问题后重检索。"""
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


def route_after_grade(state: AgentState) -> Literal["generate", "rewrite", "escalate"]:
    if state["relevant"]:
        return "generate"
    if state["rewrite_count"] < MAX_REWRITES:
        return "rewrite"
    return "escalate"


graph = StateGraph(AgentState)
graph.add_node("prepare", prepare)
graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("generate", generate)
graph.add_node("rewrite", rewrite)
graph.add_node("escalate", escalate)
graph.set_entry_point("prepare")
graph.add_edge("prepare", "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges(
    "grade",
    route_after_grade,
    {"generate": "generate", "rewrite": "rewrite", "escalate": "escalate"},
)
graph.add_edge("rewrite", "retrieve")
graph.add_edge("generate", END)
graph.add_edge("escalate", END)

support_agent = graph.compile()
