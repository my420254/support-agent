"""工具运行时（Tool Boundary）：单点执行器 + 幂等 + 错误分类 + 审计。

设计参考业界生产实践（详见 docs/TOOL_RUNTIME.md）：
- **单点执行**：模型只"提议"动作，真正的执行全部经过 ToolExecutor——
  统一做 schema 校验、风险策略、超时、重试、审计、错误归一化。
- **确定性幂等键**：写类操作必须带幂等键，且键由 (tool, args, thread_id) 确定性哈希得出。
  ⚠️ 绝不能用"每次调用随机 UUID"——真实事故：per-call UUID 导致同一笔退款被执行两次。
- **claim/complete 分离**：执行前先 claim（in_progress + lease），业务确认后才 complete(done)。
  好处：进程崩溃后能识别"进行中"而不是误判为"已完成"。
- **错误分类**："超时"意味着**结果未知**（不代表业务动作没发生），
  所以超时/网络错误要走幂等重试或查状态，不能盲目重放。

生产替换：ledger 目前是进程内字典，生产应换 Redis/Postgres（多实例共享 + 持久化）。
"""

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RiskClass(str, Enum):
    """工具风险等级——决定是否需要幂等键、是否需要人工确认。"""

    READ = "read"      # 只读：可自由重试
    WRITE = "write"    # 可逆/有界写：必须带幂等键
    ADMIN = "admin"    # 高危不可逆/特权：幂等键 + 人工确认


class ToolError(str, Enum):
    """归一化错误分类——每个类别对应不同的恢复策略。"""

    OK = "ok"
    TRANSIENT = "transient"        # 限流/网络抖动 → 指数退避重试
    NOT_FOUND = "not_found"        # 资源不存在 → 如实告知用户（不要重试）
    VALIDATION = "validation"      # 参数非法 → 修参数，不重试
    POLICY = "policy"              # 策略拒绝 → fail closed，审计
    TIMEOUT = "timeout"            # ⚠️ 结果"未知"→ 不可盲目重放，需查状态或幂等重试
    UNKNOWN = "unknown"            # 未分类 → 记录并转人工


# 可重试的错误类别（注意：TIMEOUT 不在此列——它代表"未知"，需走幂等路径）
RETRYABLE = {ToolError.TRANSIENT}
# 需要人工介入的类别
NEEDS_HUMAN = {ToolError.POLICY, ToolError.UNKNOWN}


@dataclass(frozen=True)
class ToolSpec:
    """工具契约：声明式元数据（参考 nsin08/ai_agents 的 tool contract）。"""

    name: str
    risk: RiskClass
    fn: Callable[..., Any]
    description: str = ""
    timeout_s: float = 10.0
    max_attempts: int = 3

    @property
    def needs_idempotency_key(self) -> bool:
        return self.risk in (RiskClass.WRITE, RiskClass.ADMIN)


@dataclass
class ToolResult:
    """结构化返回：agent 可据此决策（不走自然语言解析）。"""

    ok: bool
    value: Any = None
    error: ToolError = ToolError.OK
    message: str = ""
    attempts: int = 1
    elapsed_s: float = 0.0
    idempotent_hit: bool = False

    def to_text(self) -> str:
        """给模型看的结构化文本（token 高效，非散文）。"""
        if self.ok:
            return str(self.value)
        return f"[{self.error.value}] {self.message}"


@dataclass
class LedgerEntry:
    """幂等台账条目：记录一个业务动作的生命周期。"""

    key: str
    status: str  # in_progress | done | failed
    result: Any = None
    lease_until: float = 0.0
    created_at: float = field(default_factory=time.time)


class IdempotencyLedger:
    """幂等台账（进程内实现；生产换 Redis/Postgres 以支持多实例）。"""

    LEASE_S = 60.0  # 租约：超过该时间未 complete，允许重新 claim（进程可能崩了）

    def __init__(self) -> None:
        self._entries: dict[str, LedgerEntry] = {}

    def claim(self, key: str) -> tuple[bool, LedgerEntry | None]:
        """尝试认领。返回 (是否可以执行, 已有条目)。

        - 已完成 → 不执行，直接返回原结果（幂等命中）
        - 进行中且租约未过期 → 不执行（防并发重复副作用）
        - 进行中但租约已过期 → 允许重新认领（上次可能崩了）
        """
        now = time.time()
        e = self._entries.get(key)
        if e is None:
            self._entries[key] = LedgerEntry(key=key, status="in_progress", lease_until=now + self.LEASE_S)
            return True, None
        if e.status == "done":
            return False, e
        if e.lease_until > now:
            return False, e  # 有并发执行者，拒绝重复
        e.lease_until = now + self.LEASE_S  # 租约过期，重新认领
        return True, None

    def complete(self, key: str, result: Any) -> None:
        if e := self._entries.get(key):
            e.status, e.result = "done", result

    def fail(self, key: str) -> None:
        self._entries.pop(key, None)  # 失败即移除，允许后续重试


def make_idempotency_key(tool: str, args: dict, thread_id: str, step: int = 0) -> str:
    """确定性幂等键：同一业务动作在任何重试中都得到相同键。

    ⚠️ 反例（真实事故）：用 uuid4() 每次生成新键 → 重试被当成新操作 → 重复退款。
    所以这里只用 (tool, args, thread_id, step) 做哈希，不含时间戳/随机数。
    """
    payload = json.dumps(
        {"tool": tool, "args": args, "thread": thread_id, "step": step},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class ToolExecutor:
    """单点工具执行器：所有工具调用都必须经过这里。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self.ledger = IdempotencyLedger()
        self.audit: list[dict] = []  # 审计日志（生产应落盘/发 OTel）

    def register(self, spec: ToolSpec) -> None:
        # 白名单语义：未注册的工具一律拒绝（deny by default）
        self._specs[spec.name] = spec

    def execute(self, name: str, args: dict, *, thread_id: str = "", step: int = 0) -> ToolResult:
        """执行工具：校验 → 幂等 → 重试 → 审计。"""
        t0 = time.perf_counter()
        spec = self._specs.get(name)
        if spec is None:
            return self._done(name, args, t0, ToolResult(False, error=ToolError.POLICY,
                                                         message=f"未注册的工具: {name}"))

        # --- 幂等：写类操作必须有键，且用确定性键 ---
        key = None
        if spec.needs_idempotency_key:
            key = make_idempotency_key(name, args, thread_id, step)
            can_run, existing = self.ledger.claim(key)
            if not can_run and existing is not None and existing.status == "done":
                return self._done(name, args, t0, ToolResult(
                    True, value=existing.result, idempotent_hit=True))
            if not can_run:
                return self._done(name, args, t0, ToolResult(
                    False, error=ToolError.TRANSIENT, message="同一操作正在执行中，已阻止重复调用"))

        # --- 执行 + 按错误类别重试 ---
        attempts = 0
        last: ToolResult | None = None
        while attempts < spec.max_attempts:
            attempts += 1
            last = self._invoke(spec, args, attempts)
            if last.ok or last.error not in RETRYABLE:
                break
            time.sleep(min(2 ** (attempts - 1), 8) * 0.1)  # 指数退避（生产加 jitter）

        last.attempts, last.elapsed_s = attempts, time.perf_counter() - t0
        if key:
            (self.ledger.complete(key, last.value) if last.ok else self.ledger.fail(key))
        return self._done(name, args, t0, last)

    def _invoke(self, spec: ToolSpec, args: dict, attempt: int) -> ToolResult:
        """单次调用，并把异常归一化成错误分类。"""
        try:
            value = spec.fn(**args)
            return ToolResult(True, value=value, attempts=attempt)
        except TypeError as e:  # 参数不对（缺参/多余参）
            return ToolResult(False, error=ToolError.VALIDATION, message=f"参数错误: {e}", attempts=attempt)
        except TimeoutError:
            # ⚠️ 超时 = 结果未知，不代表没执行；调用方不应盲目重放
            return ToolResult(False, error=ToolError.TIMEOUT, message="执行超时，结果未知", attempts=attempt)
        except LookupError as e:
            return ToolResult(False, error=ToolError.NOT_FOUND, message=str(e), attempts=attempt)
        except PermissionError as e:
            return ToolResult(False, error=ToolError.POLICY, message=str(e), attempts=attempt)
        except Exception as e:  # 兜底
            return ToolResult(False, error=ToolError.UNKNOWN,
                              message=f"{type(e).__name__}: {e}", attempts=attempt)

    def _done(self, name: str, args: dict, t0: float, r: ToolResult) -> ToolResult:
        """记录审计日志并返回结果。"""
        self.audit.append({
            "trace_id": uuid.uuid4().hex[:12],
            "tool": name,
            "args": {k: str(v)[:60] for k, v in args.items()},  # 截断，避免敏感信息入日志
            "ok": r.ok,
            "error": r.error.value,
            "attempts": r.attempts,
            "elapsed_s": round(time.perf_counter() - t0, 4),
            "idempotent_hit": r.idempotent_hit,
        })
        return r


# 全局单例（服务层与 agent 共用）
executor = ToolExecutor()
