# 工具运行时（Tool Boundary）设计

> **重点/难点**：Agent 的可靠性主要不是"模型够不够聪明"，而是**工具执行这一层够不够稳**。
> 2026 H1 的 Agent 失败模式复盘把 "tool-call chaos" 列为三大失败簇之一：
> 参数畸形、**重试写操作却无幂等键**、工具无超时挂死、遇到模糊错误陷入死循环。

实现见 `src/agents/tool_runtime.py`；测试见 `tests/test_unit.py`。

## 1. 为什么需要单独一层

模型只"提议"动作，**真正的执行全部经过 `ToolExecutor` 这一个点**。这样以下能力集中实现，而不是散落在各工具里：

```
模型提议 → ToolExecutor ─┬─ schema 校验 / 白名单（deny by default）
                         ├─ 风险策略（read / write / admin）
                         ├─ 幂等（写类必须带确定性键）
                         ├─ 超时 + 重试（按错误类别）
                         ├─ 错误归一化
                         └─ 结构化审计
```

## 2. 三个关键设计决策（每个都有真实事故背书）

### 2.1 幂等键必须**确定性生成**，不能用随机 UUID

```python
key = sha256({"tool", "args", "thread_id", "step"})   # ✅ 确定性
key = uuid4()                                          # ❌ 每次新键 = 每次都是"新操作"
```

**依据**：真实生产事故——用 per-call UUID 时，LLM 超时后重试被当成新请求，**同一笔退款被执行两次**。
本项目的 `create_ticket` 是写操作，相同 thread + 相同参数重复调用会命中幂等、返回**原工单号**而不是新建。

### 2.2 claim / complete 分离（带租约）

执行前先 `claim`（写 `in_progress` + 60s 租约），业务确认后才 `complete(done)`：

| 台账状态 | 含义 | 行为 |
|---|---|---|
| 无记录 | 首次 | 执行 |
| `done` | 已完成 | **不执行**，直接返回原结果（幂等命中） |
| `in_progress` 且租约未过期 | 有并发执行者 | **拒绝**（防重复副作用） |
| `in_progress` 且租约已过期 | 上次可能崩了 | 允许重新认领 |

好处：进程崩溃后能区分"进行中"与"已完成"，而不是把没做完的当成做完了。

### 2.3 超时 = **结果未知**，不是"失败"

这是最容易被写错的一点：

- **失败**（明确的错误响应）→ 可以安全重试（若幂等）
- **超时**（没收到响应）→ **无法确定业务动作是否已发生** → 不能盲目重放

所以 `ToolError.TIMEOUT` **不在 `RETRYABLE` 集合里**（有单测断言这一点）。正确处置是：查状态 / 幂等重试 / 补偿 / 转人工。

## 3. 错误分类与恢复策略

| 类别 | 触发 | 策略 |
|---|---|---|
| `TRANSIENT` | 限流、网络抖动 | 指数退避重试（可重试） |
| `NOT_FOUND` | 订单不存在 | 如实告知用户，**不重试** |
| `VALIDATION` | 参数缺失/类型错 | 修参数，**不重试** |
| `POLICY` | 未注册工具 / 越权 | fail closed + 审计 |
| `TIMEOUT` | 超时 | **结果未知**，见 §2.3 |
| `UNKNOWN` | 未分类异常 | 记录 + 转人工 |

## 4. 与其他模块的关系

- `agents/tools.py`：注册工具契约（`ToolSpec`，含风险等级/超时/重试次数）
- `agents/support_agent.py`：所有工具调用走 `call_tool(...)`，不再直接调函数
- 白名单语义：**未注册的工具一律拒绝**（有单测）

## 5. 已知局限（诚实）

1. **幂等台账是进程内字典**——多实例部署需换 Redis/Postgres（否则各实例台账不共享，跨实例无法幂等）。
2. **未实现熔断器**——业界做法是按上游依赖（LLM 供应商 / 数据库）开熔断（N 次失败 → open → half-open）。当前只在工具级重试。
3. **未实现补偿事务（Saga）**——目前只有 `create_ticket` 一个写操作，无需补偿；若加入"扣款/退款"等组合写操作，需要 Saga。
4. **审计日志在内存**——生产应落盘或发 OTel（`tool_call_started/finished` 结构化事件 + trace_id）。
5. **重试退避无 jitter**——多个请求同时重试可能造成惊群。

## 6. 参考来源

- [nsin08/ai_agents — Tool Boundary (Contracts, Enforcement, MCP)](https://github.com/nsin08/ai_agents/blob/main/.context/project/agent_core_design/08_tool_boundary.md)
- [ADR-038 Reliability Taxonomy（幂等键必填、熔断参数、退避默认值）](https://github.com/BlakeMatthews-dev/maistro-engine/blob/main/docs/adr/ADR-038-reliability-taxonomy.md)
- [Agent 渡劫48关 · 第23关：Tool Timeout 后到底能不能重新执行](https://cloud.tencent.cn/developer/article/2742931)
- [FDE 进化36记 · Agent 失败以后到底怎么恢复（claim/complete 分离）](https://cloud.tencent.cn/developer/article/2742188)
- [RavenClaude — 2026 Q1/Q2 Agent 失败模式复盘](https://github.com/mcorbett51090/RavenClaude/blob/main/docs/best-practices/2026-q1-q2-failure-modes.md)
- [reliable-mcp — MCP 可靠性手册（故障注入 21 案例）](https://github.com/alexey-tyurin/reliable-mcp)
