"""加载 Dify 中文文档 → 干净文档记录，输出 JSONL。

输出 data/processed/corpus.jsonl，每行一条：
    {id, title, description, section, url, content}
其中 id = 相对路径（去 .mdx），url = docs.dify.ai 上的原文地址（供引用溯源）。
"""

import json
from collections import Counter
from pathlib import Path

from ingestion.cleaner import clean_mdx, strip_frontmatter

ROOT = Path(__file__).resolve().parents[2]  # support-agent/
RAW_DIR = ROOT / "data" / "raw" / "dify-docs" / "zh"
OUT_PATH = ROOT / "data" / "processed" / "corpus.jsonl"
BASE_URL = "https://docs.dify.ai"


def build_url(rel: str) -> str:
    """zh/cloud/use-dify/xxx.mdx -> https://docs.dify.ai/zh/cloud/use-dify/xxx"""
    p = rel.replace("\\", "/")
    if p.endswith(".mdx"):
        p = p[:-4]
    return f"{BASE_URL}/{p}"


def load_documents() -> list[dict]:
    docs: list[dict] = []
    for f in sorted(RAW_DIR.rglob("*.mdx")):
        text = f.read_text(encoding="utf-8")
        fm, body = strip_frontmatter(text)
        content = clean_mdx(body)
        if not content:
            continue
        rel = f.relative_to(RAW_DIR.parent).as_posix()  # zh/cloud/use-dify/xxx.mdx
        parts = rel.split("/")
        section = parts[1] if len(parts) > 2 else parts[0]
        doc_id = rel[:-4] if rel.endswith(".mdx") else rel
        docs.append(
            {
                "id": doc_id,
                "title": fm.get("title", ""),
                "description": fm.get("description", ""),
                "section": section,
                "url": build_url(rel),
                "content": content,
            }
        )
    return docs


def main() -> None:
    docs = load_documents()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")

    total_chars = sum(len(d["content"]) for d in docs)
    print(f"loaded {len(docs)} docs, {total_chars} chars total")
    print("sections:")
    for sec, n in Counter(d["section"] for d in docs).most_common():
        print(f"  {sec}: {n}")


if __name__ == "__main__":
    main()
