"""可靠性验收：crash-resume（中断跨进程恢复）+ session isolation（多会话记忆隔离）。

两个都是"生产级"区别于"demo"的关键：
1. crash-resume：interrupt 挂起的状态必须持久化到 checkpoint，模拟重启（新进程）后能 resume；
2. session isolation：长期记忆必须按 user_id 隔离，用户 B 绝不能读到用户 A 的订单号。

用法：python scripts/reliability_check.py
"""

import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from starlette.testclient import TestClient  # noqa: E402

from service import app  # noqa: E402


def check_crash_resume() -> bool:
    """投诉 → interrupt 挂起 → 新进程（新 TestClient 同 SQLite）→ resume。"""
    with TestClient(app) as c1:
        r1 = c1.post("/invoke", json={"message": "我要投诉，服务太差了", "thread_id": "crash-1", "user_id": "u1"})
        assert "转接人工" in r1.json().get("content", ""), f"第一次应挂起: {r1.json()}"
    with TestClient(app) as c2:  # 模拟进程重启（全新进程，同一 checkpoints.db）
        r2 = c2.post("/invoke", json={"message": "人工客服：已为您退款", "thread_id": "crash-1", "user_id": "u1"})
        assert "人工客服回复" in r2.json().get("content", ""), f"重启后应能 resume: {r2.json()}"
    return True


def check_session_isolation() -> bool:
    """用户 A 记住订单号后，用户 B 查订单不得拿到 A 的订单。"""
    with TestClient(app) as c:
        c.post("/invoke", json={"message": "查订单 ORD12345", "thread_id": "a1", "user_id": "userA"})
        rb = c.post("/invoke", json={"message": "帮我查一下我的订单", "thread_id": "b1", "user_id": "userB"})
        content = rb.json().get("content", "")
        # 泄漏判定：B 若拿到了 A 的实际订单结果（含商品信息），即为泄漏
        leaked = "商品「" in content
        assert not leaked, f"会话隔离失败：用户B 读到了 A 的记忆: {content}"
    return True


if __name__ == "__main__":
    print("① crash-resume:", "PASS" if check_crash_resume() else "FAIL")
    print("② session isolation:", "PASS" if check_session_isolation() else "FAIL")
