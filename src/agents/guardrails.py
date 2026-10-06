"""安全护栏：Prompt Injection 检测 + PII 脱敏。

1. PII 脱敏：手机号/邮箱/身份证号在进入 LLM 与工具前打码，防止泄露与日志污染。
2. Prompt Injection 检测：正则匹配常见越狱/注入话术（"忽略之前的指令"、"扮演…"等），
   命中直接转人工，不让注入内容影响检索/工具/生成。
"""

import re

PHONE_RE = re.compile(r"1[3-9]\d{9}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w]+")
ID_RE = re.compile(r"\b\d{17}[\dXx]\b")

INJECTION_PATTERNS = [
    r"忽略.{0,8}(之前|以上|上面)的.{0,8}(指令|规则|要求|提示)",
    r"ignore.{0,12}(previous|above|all).{0,12}(instructions|prompt|rules)",
    r"(你现在|请|假装).{0,10}(扮演|角色|是).{0,10}(一个|一位)",
    r"system\s*prompt",
    r"(泄露|泄漏|输出).{0,6}(系统|提示词|prompt)",
]


def mask_pii(text: str) -> str:
    """脱敏手机号/邮箱/身份证号。"""
    text = PHONE_RE.sub("[手机号]", text)
    text = EMAIL_RE.sub("[邮箱]", text)
    text = ID_RE.sub("[证件号]", text)
    return text


def detect_prompt_injection(text: str) -> bool:
    """检测常见的 prompt 注入话术。"""
    return any(re.search(p, text, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS)
