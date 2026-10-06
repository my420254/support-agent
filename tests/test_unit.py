"""核心纯函数的单元测试（不依赖 LLM / Qdrant）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.metrics import hit_at_k, reciprocal_rank, unique_doc_ids
from ingestion.chunker import chunk_markdown
from ingestion.cleaner import clean_mdx, strip_frontmatter


def test_strip_frontmatter():
    fm, body = strip_frontmatter('---\ntitle: "部署"\ndescription: "d"\n---\n正文内容')
    assert fm == {"title": "部署", "description": "d"}
    assert body.strip() == "正文内容"


def test_clean_mdx_strips_jsx_and_links():
    text = "<Info>\n点击 [这里](/zh/guide)\n</Info>\n![图](/images/x.png)"
    out = clean_mdx(text)
    assert "这里" in out
    assert "/zh/guide" not in out  # 链接 URL 被去掉
    assert "图" in out  # 图片 alt 保留
    assert "<Info>" not in out


def test_chunker_keeps_code_block_intact():
    content = "## 部署\n\n```python\n## 这是注释不是标题\nprint('hi')\n```\n\n### 步骤\n正文"
    chunks = chunk_markdown(content)
    # 代码块里的 ## 注释不应被当成标题切断
    assert any("## 这是注释不是标题" in c["text"] for c in chunks)


def test_chunker_splits_on_headings():
    content = "## A\n内容A\n\n## B\n内容B"
    chunks = chunk_markdown(content)
    headings = {c["heading"] for c in chunks}
    assert "A" in headings and "B" in headings


def test_metrics_hit_and_mrr():
    assert hit_at_k("doc1", ["doc1", "doc2"], 1) == 1
    assert hit_at_k("doc1", ["doc2", "doc1"], 1) == 0
    assert reciprocal_rank("doc1", ["doc2", "doc1", "doc3"]) == 0.5
    assert unique_doc_ids([{"doc_id": "a"}, {"doc_id": "a"}, {"doc_id": "b"}]) == ["a", "b"]
