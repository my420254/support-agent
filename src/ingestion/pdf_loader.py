"""PDF 解析：文本 + 表格 + **分栏感知**。

为什么需要专门实现（而不是直接 pypdf.extract_text）：
1. **双栏版面**：学术论文/报告常是两栏，朴素抽取会把左右栏文字交错混在一起，
   句子读不通、检索质量崩。这里先检测是否双栏，是则按左右两半分别抽取再拼接。
2. **表格**：用 pdfplumber 抽表并转成 markdown 表格，保留结构（否则行列关系全丢）。
3. **页码元数据**：保留页码，供引用溯源（"见第 12 页"）。

依赖：pypdf（文本）+ pdfplumber（分栏/表格）。
"""

from pathlib import Path

import pdfplumber
from pypdf import PdfReader

MID_BAND = (0.42, 0.58)  # 页宽中间带（用于判断是否双栏）
CROSS_RATIO = 0.15  # 中间带内词占比超过该值 → 视为单栏


def _is_two_column(page) -> bool:
    """启发式判断双栏：统计**字符**的 x 中心落在页面中间带的占比。

    用字符而非"词"：部分 PDF 缺空格导致 extract_words 返回的词数极少，
    按词统计会失效（实测某学术论文每页仅数个"词"）。字符级统计更鲁棒。
    """
    try:
        chars = page.chars
    except Exception:
        return False
    if len(chars) < 200:  # 字符太少（封面/纯图页）不做分栏处理
        return False
    w = page.width
    lo, hi = w * MID_BAND[0], w * MID_BAND[1]
    cross = sum(1 for c in chars if lo <= (c["x0"] + c["x1"]) / 2 <= hi)
    return (cross / len(chars)) < CROSS_RATIO


def _extract_page(page) -> str:
    """抽取单页文本；双栏则先左后右，避免左右交错。"""
    if _is_two_column(page):
        w, h = page.width, page.height
        left = page.crop((0, 0, w * 0.52, h)).extract_text() or ""
        right = page.crop((w * 0.48, 0, w, h)).extract_text() or ""
        return (left.strip() + "\n" + right.strip()).strip()
    return (page.extract_text() or "").strip()


def _tables_to_markdown(page) -> list[str]:
    """把该页的表格转成 markdown 表格（保留行列结构）。"""
    out: list[str] = []
    try:
        tables = page.extract_tables()
    except Exception:
        return out
    for tb in tables:
        rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in tb if row]
        if len(rows) < 2:
            continue
        header = "| " + " | ".join(rows[0]) + " |"
        sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
        out.append("\n".join([header, sep, *body]))
    return out


def load_pdf(path: str | Path, title: str | None = None) -> dict:
    """把 PDF 解析成与 MDX 语料同构的一条记录。

    返回 {id, title, description, section, url, content}，
    content 按页组织（`## 第 N 页`），表格以 markdown 形式内嵌。
    """
    path = Path(path)
    reader = PdfReader(str(path))
    n_pages = len(reader.pages)

    parts: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = _extract_page(page)
            tables = _tables_to_markdown(page)
            block = f"## 第 {i} 页\n{text}"
            if tables:
                block += "\n\n" + "\n\n".join(tables)
            parts.append(block)

    content = "\n\n".join(parts)
    return {
        "id": f"pdf/{path.stem}",
        "title": title or path.stem,
        "description": f"PDF 文档，共 {n_pages} 页",
        "section": "pdf",
        "url": str(path),
        "content": content,
    }


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    p = sys.argv[1]
    doc = load_pdf(p)
    print(f"解析完成: {doc['title']} | {len(doc['content'])} 字符")
    print(doc["content"][:600])
