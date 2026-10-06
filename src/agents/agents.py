from dataclasses import dataclass

from langgraph.graph.state import CompiledStateGraph

from agents.support_agent import support_agent
from schema import AgentInfo

DEFAULT_AGENT = "support-agent"

AgentGraph = CompiledStateGraph


@dataclass
class Agent:
    description: str
    graph: AgentGraph


agents: dict[str, Agent] = {
    "support-agent": Agent(
        description="企业知识库智能客服：Agentic RAG + 工具调用 + 转人工",
        graph=support_agent,
    ),
}


async def load_agent(agent_id: str) -> None:
    """All agents are eager-loaded; kept for API compatibility with the service layer."""
    return None


def get_agent(agent_id: str) -> AgentGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
