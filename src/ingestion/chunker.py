"""把清洗后的文档切成语义块（标题感知）。

策略：按 ## / ### 标题把文档切成 section；section 超过 chunk_size 时硬切并带 overlap。
每块保留 heading（所属小节标题），供检索时拼接上下文、以及生成引用。
"""

from pathlib import Path


def _fixed_chunks(content: str, chunk_size: int, overlap: int) -> list[dict]:
    """朴素定长切分（忽略标题结构），作为消融基线。"""
    chunks: list[dict] = []
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(content):
        piece = content[start : start + chunk_size].strip()
        if piece:
            chunks.append({"heading": "", "text": piece})
        if start + chunk_size >= len(content):
            break
        start += step
    return chunks


def chunk_markdown(
    content: str, chunk_size: int = 400, overlap: int = 60, strategy: str = "heading"
) -> list[dict]:
    """返回 [{heading, text}]。

    strategy:
      - "heading"（默认）：标题感知，按 ## / ### 切 section，超长再硬切；
      - "fixed"：朴素定长切分（忽略标题），用于消融对比。
    """
    if strategy == "fixed":
        return _fixed_chunks(content, chunk_size, overlap)

    sections: list[tuple[str, str]] = []
    cur_heading = ""
    cur_lines: list[str] = []
    in_code_block = False

    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            cur_lines.append(line)
            continue

        # 仅在代码块外部识别合法的 Markdown 标题（# ~ ####，必须带空格）
        if not in_code_block and (
            line.startswith("## ") or line.startswith("### ") or line.startswith("#### ") or line.startswith("# ")
        ):
            if cur_lines:
                sections.append((cur_heading, "\n".join(cur_lines)))
            cur_heading = line.lstrip("#").strip()
            cur_lines = []
        else:
            cur_lines.append(line)

    if cur_lines:
        sections.append((cur_heading, "\n".join(cur_lines)))

    chunks: list[dict] = []
    step = max(1, chunk_size - overlap)
    for heading, body in sections:
        body = body.strip()
        if not body:
            continue
        if len(body) <= chunk_size:
            chunks.append({"heading": heading, "text": body})
        else:
            start = 0
            while start < len(body):
                piece = body[start : start + chunk_size].strip()
                if piece:
                    chunks.append({"heading": heading, "text": piece})
                if start + chunk_size >= len(body):
                    break
                start += step
    return chunks

