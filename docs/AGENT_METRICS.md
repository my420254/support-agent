# Agent 工程指标（设计 + 定义）

> RAG 侧的指标（Hit@k / MRR / faithfulness）衡量"检索准不准"。
> 本文档定义 **Agent 侧**指标：衡量"任务有没有完成、路径对不对、稳不稳、快不快"。
> 实现见 `src/eval/agent_eval.py`，任务集见 `data/golden/agent_tasks.jsonl`。

## 1. 设计原则

1. **主指标是端到端 Task Success，不是组件准确率。**
   路由对 + 工具对 + 参数对，**不代表任务完成**（例：工具返回 `未找到订单`，Agent 却答"正在配送"）。
   组件指标降为**归因手段**——失败了再回溯是哪一环错的。

2. **评测 expected outcome 约束，而非 exact trajectory。**
   合法路径可能不唯一（如未来 Agent 先查记忆再查订单也正确）。
   任务集只声明约束：`route` / `entities` / `allowed_tools` / `handoff` / `success`。

3. **埋点不侵入业务代码。**
   全部通过 LangGraph `astream` 的事件流采集，不往 agent 节点里塞计时逻辑。

## 2. 指标定义

| 指标 | 计算方式 | 说明 |
|---|---|---|
| **Task Success Rate**（主） | 按 `success` 约束判定终态 | 例：`order_status_returned` → 终答含订单状态；`injection_blocked` → 意图=escalate 且未建工单 |
| Route Accuracy | 实际意图 == 期望 route | 归因用 |
| Action Accuracy | 实际执行的节点集合 == 期望工具映射的节点集合 | 归因用（当前架构"工具"是节点内函数） |
| **Handoff Precision** | tp/(tp+fp) | 不该转人工时**没乱转**吗 |
| **Handoff Recall** | tp/(tp+fn) | 该转人工时**转了**吗 |
| Avg Steps / Task | 节点执行次数均值 | 是否绕路 |
| **TTFT P50/P95** | 任务开始 → 首个流式 token | 用户体验第一指标 |
| E2E P50/P95 | 任务开始 → 完成 | SLA |

## 3. 任务集设计（47 题，分层覆盖）

| 类别 | 数量 | 覆盖 |
|---|---|---|
| order（带单号） | 6 | 工具调用 + 结果利用 |
| order（无单号） | 2 | 缺参数时反问 |
| kb（知识问答） | 15 | 路由 + 检索 + 生成 |
| chitchat | 5 | 短路 |
| complaint | 4 | 建工单 + HITL 中断 |
| escalate（越界） | 6 | 正确拒答 |
| injection | 4 | 安全护栏拦截 |
| PII | 1 | 脱敏后仍正确路由 |

## 4. 已知局限

- **LLM-as-judge 未校准**：`success` 判定目前用结构化约束 + 少量字符串规则，**没有**人工标注的校准集，也没报 Cohen's κ。
- **单轮为主**：多轮有状态任务（Turn1 给单号 → Turn2 追问 → Turn3 申请售后）**尚未覆盖**，这是下一步。
- **pass^k 未做**：k 次重复运行的一致性（Agent 不稳定性）尚未测量。
- **fault injection 未做**：LLM 超时/工具报错/重复 resume 等故障注入测试待补。
- 样本量 47，**统计置信区间仍偏宽**（±14pp @95%），结论应作为方向性参考。
