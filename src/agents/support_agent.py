"""企业知识库智能客服 Agent（P0 骨架）。

P0 只验证「模型 → 工具 → 模型」的工具调用循环能跑通（DeepSeek function calling）。
P3 会替换成完整的 Agentic RAG 图：
    路由 → 检索 → 相关性评分 → 纠正式改写 → 工具 → 带引用生成 → 自检 → 转人工
"""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda, RunnableSerializable
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agents.tools import lookup_knowledge_base
from core import get_model, settings

INSTRUCTIONS = """你是「Acme 云服务」的智能客服，负责回答用户关于产品使用、账户、账单、政策的问题。

规则：
1. 涉及产品功能、政策、操作步骤时，先调用 lookup_knowledge_base 查询知识库，再基于返回内容回答，并标注来源。
2. 知识库查不到、或问题超出客服职责范围时，明确告知用户你无法回答并建议转人工，不要编造。
3. 用中文简洁回答。
"""


class AgentState(MessagesState, total=False):
    pass


tools = [lookup_knowledge_base]


def wrap_model(model: BaseChatModel) -> RunnableSerializable[AgentState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=INSTRUCTIONS)] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model  # type: ignore[return-value]


async def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
    m = get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))
    model_runnable = wrap_model(m)
    response = await model_runnable.ainvoke(state, config)
    return {"messages": [response]}


def route_after_model(state: AgentState) -> Literal["tools", "__end__"]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "__end__"


agent = StateGraph(AgentState)
agent.add_node("model", call_model)
agent.add_node("tools", ToolNode(tools))
agent.set_entry_point("model")
agent.add_conditional_edges("model", route_after_model, {"tools": "tools", "__end__": END})
agent.add_edge("tools", "model")

support_agent = agent.compile()
