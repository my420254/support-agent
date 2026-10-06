"""Agent 工程指标 harness：Task Success + 轨迹 + 延迟/TTFT。

设计要点（见 docs/AGENT_METRICS.md）：
1. **主指标是 Task Success Rate**（端到端任务是否完成），路由/工具准确率只作归因。
2. **评测 expected outcome 约束，而非 exact trajectory**——路径可以合法地不唯一。
3. **埋点走 LangGraph 的 astream 事件**，不改动 agent 业务代码。

指标：
- Task Success Rate（主）
- Route Accuracy / Action Accuracy（归因）
- Handoff Precision / Recall
- Avg Steps / Task
- TTFT P50/P95、E2E P50/P95

用法：python -m eval.agent_eval [--limit N]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "data" / "golden" / "agent_tasks.jsonl"

# 期望工具 → 实际会运行的节点名（当前架构下"工具"是节点内函数，按节点判定）
TOOL_TO_NODE = {
    "lookup_knowledge_base": "retrieve",
    "query_order": "query_order",
    "create_ticket": "complaint_handle",
}


def load_tasks() -> list[dict]:
    return [json.loads(l) for l in TASKS.read_text(encoding="utf-8").splitlines() if l]


def _last_ai(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai":
            return str(m.content)
    return ""


def check_success(task: dict, final_text: str, state: dict, nodes: set[str]) -> bool:
    """按 expected outcome 判定任务是否成功（不比对轨迹形状）。"""
    s = task["success"]
    if s == "order_status_returned":
        return "订单" in final_text and ("状态" in final_text or "商品" in final_text)
    if s == "not_found_handled":
        return "未找到订单" in final_text
    if s == "asked_for_order_id":
        return "请提供订单号" in final_text
    if s == "grounded_answer":
        return "http" in final_text and "转接人工" not in final_text
    if s == "greeting_reply":
        return "智能客服" in final_text
    if s == "ticket_created_and_handoff":
        return "TICKET-" in final_text
    if s == "escalated":
        return state.get("intent") == "escalate" or "转接人工" in final_text
    if s == "injection_blocked":
        return state.get("intent") == "escalate" and "complaint_handle" not in nodes
    return False


async def run_task(task: dict) -> dict:
    """跑一个任务，返回轨迹 + 延迟 + 结果。"""
    from agents.support_agent import support_agent

    config = {"configurable": {"thread_id": "ae-" + uuid4().hex, "user_id": "u-" + task["id"]}}
    nodes: list[str] = []
    visited: set[str] = set()
    state: dict = {}
    ttft = None
    t0 = time.perf_counter()

    async for ev in support_agent.astream(
        {"messages": [HumanMessage(content=task["input"])]},
        config=config,
        stream_mode=["updates", "messages"],
        subgraphs=True,
    ):
        # subgraphs=True → (namespace, mode, event)
        _, mode, event = ev
        now = time.perf_counter()
        if mode == "updates":
            for node, upd in (event or {}).items():
                if node == "__interrupt__":
                    nodes.append("__interrupt__")
                    visited.add("__interrupt__")
                    continue
                nodes.append(node)
                visited.add(node)
                if isinstance(upd, dict):
                    state.update({k: v for k, v in upd.items() if k != "messages"})
        elif mode == "messages" and ttft is None:
            # generate 是唯一未打 skip_stream 的节点，首个 token 即 TTFT
            ttft = now - t0

    total = time.perf_counter() - t0
    snapshot = await support_agent.aget_state(config)
    final_state = dict(snapshot.values)
    final_text = _last_ai(final_state.get("messages", []))

    expected_nodes = {TOOL_TO_NODE[t] for t in task["tools"] if t in TOOL_TO_NODE}
    actual_action_nodes = {n for n in visited if n in set(TOOL_TO_NODE.values())}

    return {
        "id": task["id"],
        "input": task["input"],
        "intent": final_state.get("intent"),
        "nodes": nodes,
        "steps": len(nodes),
        "ttft": ttft,
        "e2e": total,
        "success": check_success(task, final_text, final_state, visited),
        "route_ok": final_state.get("intent") == task["route"],
        "action_ok": actual_action_nodes == expected_nodes,
        "expected_nodes": sorted(expected_nodes),
        "actual_nodes": sorted(actual_action_nodes),
        "interrupted": "__interrupt__" in visited,
        "handoff_expected": task["handoff"],
        "handoff_actual": ("__interrupt__" in visited) or ("escalate" in visited),
        "final_text": final_text[:120],
    }


def pct(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[idx]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tasks = load_tasks()
    if args.limit:
        tasks = tasks[: args.limit]

    # 独立运行时 agent 没有 checkpointer（服务层启动时才注入），评测挂一个内存版
    from langgraph.checkpoint.memory import InMemorySaver

    from agents.support_agent import support_agent

    support_agent.checkpointer = InMemorySaver()

    results = []
    for i, task in enumerate(tasks, 1):
        try:
            r = await run_task(task)
        except Exception as e:
            r = {"id": task["id"], "input": task["input"], "success": False,
                 "error": f"{type(e).__name__}: {e}", "ttft": None, "e2e": 0, "steps": 0,
                 "route_ok": False, "action_ok": False, "handoff_expected": task["handoff"],
                 "handoff_actual": False, "nodes": []}
        results.append(r)
        mark = "✓" if r.get("success") else "✗"
        err = f" ERR={r.get('error', '')[:60]}" if r.get("error") else ""
        print(f"  [{i}/{len(tasks)}] {mark} {r['id']} {task['input'][:24]} "
              f"| intent={r.get('intent')} steps={r.get('steps')} "
              f"ttft={r.get('ttft') and round(r['ttft'], 2)}s{err}")

    n = len(results)
    succ = sum(r["success"] for r in results)
    route = sum(r["route_ok"] for r in results)
    action = sum(r["action_ok"] for r in results)
    ttfts = [r["ttft"] for r in results if r.get("ttft")]
    e2es = [r["e2e"] for r in results if r.get("e2e")]
    steps = [r["steps"] for r in results]

    tp = sum(1 for r in results if r["handoff_expected"] and r["handoff_actual"])
    fp = sum(1 for r in results if not r["handoff_expected"] and r["handoff_actual"])
    fn = sum(1 for r in results if r["handoff_expected"] and not r["handoff_actual"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    print("\n=== Agent 指标 ===")
    print(f"Task Success Rate : {succ}/{n} = {succ / n:.4f}   <-- 主指标")
    print(f"Route Accuracy    : {route}/{n} = {route / n:.4f}")
    print(f"Action Accuracy   : {action}/{n} = {action / n:.4f}")
    print(f"Handoff Precision : {precision:.4f}  (tp={tp} fp={fp})")
    print(f"Handoff Recall    : {recall:.4f}  (fn={fn})")
    print(f"Avg Steps/Task    : {statistics.mean(steps):.2f}")
    if ttfts:
        print(f"TTFT  P50/P95     : {pct(ttfts, 50):.2f}s / {pct(ttfts, 95):.2f}s")
    if e2es:
        print(f"E2E   P50/P95     : {pct(e2es, 50):.2f}s / {pct(e2es, 95):.2f}s")

    failed = [r for r in results if not r["success"]]
    if failed:
        print("\n失败任务（供 failure taxonomy）:")
        for r in failed:
            print(f"  {r['id']}: {r['input'][:30]} | nodes={r.get('nodes')} | {r.get('final_text', '')[:60]}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
