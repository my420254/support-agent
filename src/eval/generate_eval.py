"""生成端评测：转人工准确率 + 引用准确率 + 幻觉率（faithfulness）。

对应 DESIGN.md §8 的「客服特有指标」。检索指标看"找没找对"，这里看"答没答对、编没编造"。

1. 转人工/闲聊准确率：负样本（投诉/越界/闲聊）是否被正确短路，而非编造 KB 答案。
2. 引用准确率：回答里引用的 URL 是否都来自检索到的文档（杜绝"编引用"）。
3. 幻觉率：LLM judge 判断回答是否完全基于上下文（1 - faithfulness）。

用法：python -m eval.generate_eval
"""

import asyncio
import json
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.support_agent import support_agent
from core import get_model, settings

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "golden" / "golden.jsonl"
NEGATIVE = ROOT / "data" / "golden" / "negative.jsonl"

URL_RE = re.compile(r"https?://[^\s\)\]），。]+")

FAITHFUL_PROMPT = """判断下面的回答是否完全基于给定的文档上下文（没有编造文档之外的事实）。

文档上下文：
{documents}

回答：
{answer}

只输出一个词：YES（回答完全基于上下文，无编造）或 NO（回答包含编造/上下文之外的内容）。"""


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]


async def _run(q: str) -> dict:
    config = {"configurable": {"thread_id": "ge-" + q[:6]}}
    return await support_agent.ainvoke({"messages": [HumanMessage(content=q)]}, config=config)


def _last_ai(messages: list) -> str:
    for m in reversed(messages):
        if getattr(m, "type", "") == "ai":
            return str(m.content)
    return ""


def _cited_urls(answer: str) -> list[str]:
    return URL_RE.findall(answer)


async def eval_transfer(negatives: list[dict]) -> tuple[int, int]:
    """转人工/闲聊准确率。"""
    ok = 0
    for g in negatives:
        result = await _run(g["question"])
        intent = result.get("intent", "")
        expected = g["expected"]
        if expected == intent:
            ok += 1
        elif expected == "escalate" and "转接人工" in _last_ai(result["messages"]):
            # route 判成 kb 但 corrective 最终诚实转人工，也算对
            ok += 1
    return ok, len(negatives)


async def eval_answerable(items: list[dict]) -> tuple[int, int, int, int]:
    """对可回答问题：引用准确率（机械）+ 幻觉率（LLM judge）。"""
    cite_ok = cite_total = 0
    faithful_ok = faithful_total = 0
    model = get_model(settings.DEFAULT_MODEL)
    for g in items:
        result = await _run(g["question"])
        answer = _last_ai(result["messages"])
        docs = result.get("documents", [])
        doc_urls = {d["url"] for d in docs}

        for u in _cited_urls(answer):
            cite_total += 1
            if u in doc_urls:
                cite_ok += 1

        ctx = "\n\n".join(f"{d['title']}: {d['text'][:300]}" for d in docs)
        resp = await model.ainvoke(
            [SystemMessage(FAITHFUL_PROMPT.format(documents=ctx, answer=answer[:2000]))]
        )
        faithful_total += 1
        if str(resp.content).strip().lower().startswith("yes"):
            faithful_ok += 1
    return cite_ok, cite_total, faithful_ok, faithful_total


async def main() -> None:
    negatives = _load(NEGATIVE)
    transfer_ok, transfer_n = await eval_transfer(negatives)

    items = _load(GOLDEN)[:15]  # 子集，控制 LLM judge 耗时
    cite_ok, cite_total, faithful_ok, faithful_n = await eval_answerable(items)

    print("=== 生成端指标 ===")
    print(f"转人工/闲聊准确率: {transfer_ok}/{transfer_n} = {transfer_ok / transfer_n:.4f}")
    if cite_total:
        print(f"引用准确率: {cite_ok}/{cite_total} = {cite_ok / cite_total:.4f}")
    else:
        print("引用准确率: 无引用（N/A）")
    print(f"幻觉率(1-faithful): {1 - faithful_ok / faithful_n:.4f}  (faithful {faithful_ok}/{faithful_n})")


if __name__ == "__main__":
    asyncio.run(main())
