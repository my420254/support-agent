"""坏例挖掘（bad case mining）——让测评集"活"起来的机制。

不是一次性做完 120 题就固定，而是：
1. 对 golden 集跑检索，找出"正确文档不在 top5"的失败题（真正难/有问题的 case）；
2. 归档到 badcases.jsonl 供人工复核（判断是题目有问题，还是检索有真缺口）；
3. 复核后回填 golden 集 → 越用越大、越用越难，指标随迭代持续变严。

用法：python -m eval.mine_badcases
"""

import json
import sys
from pathlib import Path

from agents.tools import search_knowledge_base_hybrid
from eval.metrics import unique_doc_ids

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "data" / "golden" / "golden.jsonl"
BADCASES = ROOT / "data" / "golden" / "badcases.jsonl"


def main() -> None:
    golden = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l]
    hits = {k: 0 for k in (1, 3, 5)}
    bad: list[dict] = []

    for g in golden:
        results = search_knowledge_base_hybrid(g["question"], top_k=5)
        doc_ids = unique_doc_ids(results)
        for k in hits:
            if g["gt_doc_id"] in doc_ids[:k]:
                hits[k] += 1
        # 正确文档完全没进 top5 = 坏例，归档复核
        if g["gt_doc_id"] not in doc_ids[:5]:
            bad.append(
                {
                    "id": g["id"],
                    "question": g["question"],
                    "gt_doc_id": g["gt_doc_id"],
                    "gt_title": g["title"],
                    "top5_docs": doc_ids,
                }
            )

    n = len(golden)
    print(f"golden {n} 题 | Hit@1 {hits[1] / n:.3f} | Hit@3 {hits[3] / n:.3f} | Hit@5 {hits[5] / n:.3f}")
    print(f"坏例（gt 未进 top5）: {len(bad)} 题")

    if bad:
        BADCASES.write_text(
            "\n".join(json.dumps(b, ensure_ascii=False) for b in bad) + "\n", encoding="utf-8"
        )
        print(f"已归档 -> {BADCASES}（人工复核后回填 golden 集）")


if __name__ == "__main__":
    main()
