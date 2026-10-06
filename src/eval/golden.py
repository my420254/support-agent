"""构建评估集（golden set）。

方法：grounded generation（业界冷启动 golden set 标准做法，见 DESIGN.md §8）。
从语料分层采样 N 篇文档，对每篇生成一个「能用该文档回答」的客服问题，
ground truth = 源文档 id（用于检索评测 Hit@k / MRR）。

诚实说明：问题是模型从真实文档生成的（来源可追溯），KB 语料是真实文档；
生成的问题会抽样人工核对。这是没有真实线上流量时的标准冷启动做法。
"""

import json
import random
import sys
import time
from pathlib import Path

from langchain_core.messages import SystemMessage

from core import get_model, settings

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"
GOLDEN = ROOT / "data" / "golden" / "golden.jsonl"

GEN_PROMPT = """你是客服问题构造器。基于下面的产品文档，生成一个真实、口语化的中文用户问题（像用户真的会向客服问的那样）。

要求：
- 这个问题必须能用下面这段文档回答。
- 涉及文档里的具体细节（功能、配置、步骤、报错等），不要太泛。
- 只输出一个问题，一行，不要解释、不要编号、不要引号。

文档标题：{title}
文档分区：{section}

文档内容（节选）：
{content}"""


def load_docs() -> list[dict]:
    return [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines() if l]


def sample_docs(docs: list[dict], n: int = 120, seed: int = 42) -> list[dict]:
    """分层采样：按 section 大小比例抽取，大分区更多题，保证覆盖各分区。"""
    rng = random.Random(seed)
    by_section: dict[str, list[dict]] = {}
    for d in docs:
        by_section.setdefault(d["section"], []).append(d)

    sampled: list[dict] = []
    for sec, ds in sorted(by_section.items()):
        k = max(1, round(n * len(ds) / len(docs)))
        rng.shuffle(ds)
        sampled.extend(ds[:k])
    rng.shuffle(sampled)
    return sampled[:n]


def pick_content(doc: dict, max_chars: int = 1500) -> str:
    return doc["content"][:max_chars]


def generate() -> None:
    docs = load_docs()
    sampled = sample_docs(docs)
    print(f"sampled {len(sampled)} docs from {len(docs)}")

    model = get_model(settings.DEFAULT_MODEL)
    golden: list[dict] = []
    for i, doc in enumerate(sampled):
        prompt = GEN_PROMPT.format(
            title=doc["title"], section=doc["section"], content=pick_content(doc)
        )
        try:
            resp = model.invoke([SystemMessage(prompt)])
            q = str(resp.content).strip().strip('"').strip("'").strip()
            golden.append(
                {
                    "id": f"q{i:03d}",
                    "question": q,
                    "gt_doc_id": doc["id"],
                    "section": doc["section"],
                    "title": doc["title"],
                }
            )
            print(f"  [{i+1}/{len(sampled)}] {q[:45]}")
        except Exception as e:
            print(f"  [{i+1}/{len(sampled)}] FAILED: {e}")
        time.sleep(0.3)

    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN.open("w", encoding="utf-8") as fh:
        for g in golden:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"saved {len(golden)} questions -> {GOLDEN}")


if __name__ == "__main__":
    generate()
