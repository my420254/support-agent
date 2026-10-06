"""MDX 文档 → 干净 markdown 文本。

Dify 文档是 Mintlify MDX：YAML frontmatter + markdown + JSX 组件（<Info>/<Frame>/<img> 等）。
这里剥掉 JSX、图片/链接语法、自动翻译提示，保留标题与正文结构，供检索与分块使用。
"""

import re

# JSX 组件标签（大写开头，成对或自闭合）：<Info> </Info> <Frame> <Tabs> <br/> <img .../>
_TAG_RE = re.compile(r"</?[A-Z][^>]*?/?>")

# markdown 图片 ![alt](url) → 保留 alt
_IMG_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")

# markdown 链接 [text](url) → 保留 text
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")

# 官方机翻提示行
_TRANSLATE_NOTE_RE = re.compile(r">\s*本文档由 AI 自动翻译[^\n]*")


def strip_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """返回 (frontmatter dict, 正文)。frontmatter 形如 key: "value"。"""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    fm_text = text[3:end]
    body = text[end + 4 :]
    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        fm[key.strip()] = value.strip().strip('"').strip("'")
    return fm, body


def clean_mdx(text: str) -> str:
    """在代码块之外清理 JSX/图片/链接，返回干净 markdown。"""
    out: list[str] = []
    in_code = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if _TRANSLATE_NOTE_RE.match(line.strip()):
            continue
        line = _TAG_RE.sub("", line)
        line = _IMG_RE.sub(r"\1", line)
        line = _LINK_RE.sub(r"\1", line)
        out.append(line)

    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
