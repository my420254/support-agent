# 项目评估报告（供外部评审）

> 本文档面向**技术评审者**：完整说明项目做了什么、各模块职责、全部真实指标、关键设计决策的理由、以及**已知短板与推荐改进**。目标是让评审者在不读代码的情况下也能判断项目深度，并快速定位可深入追问的点。

---

## 一、项目概况

| 项 | 内容 |
|---|---|
| 定位 | 企业知识库智能客服 Agent（Agentic RAG），面向真实落地 |
| 语料 | Dify 官方中文文档 **291 篇 / 98 万字符 → 4043 chunks** |
| 技术栈 | LangGraph + FastAPI + Qdrant + DeepSeek API + bge-small-zh（embedding）+ SQLite/Postgres + Langfuse |
| 代码规模 | 29 commits，48 个受版本控制文件 |
| 测试 | 7 个纯函数单测（chunker/metrics/cleaner/guardrails）+ 2 个可靠性验收脚本 |

**它要证明的不是"能回答问题"，而是**：有真实指标、真实调优记录、真实故障排查、可观测、可容器化的工程能力。

---

## 二、模块清单

| 层 | 模块 | 职责 | 关键设计点 |
|---|---|---|---|
| **数据管线** | `ingestion/cleaner.py` | MDX → 干净 markdown | 剥 JSX 组件/图片/链接，**代码块内不误处理** |
| | `ingestion/loader.py` | 建元数据 | 标题/分区/**原文 URL**（引用溯源用） |
| | `ingestion/chunker.py` | 标题感知分块 | 跟踪 ` ``` `/`~~~` 代码围栏，避免代码内 `##` 注释被当标题腰斩；防 overlap≥chunk_size 死循环 |
| **检索** | `retrieval/embedder.py` | 向量化 | bge-small-zh（512 维）；离线加载（`HF_HUB_OFFLINE`） |
| | `retrieval/qdrant_store.py` | 向量库封装 | 用 `127.0.0.1` 规避 Windows IPv6 慢路径 |
| | `retrieval/hybrid.py` | **混合检索** | dense + BM25(jieba) → **weighted RRF**（α=0.1） |
| | `retrieval/reranker.py` | cross-encoder 重排 | bge-reranker-v2-m3（**实测后决定不采用，见 §5**） |
| **Agent** | `agents/guardrails.py` | 安全护栏 | Prompt Injection 检测 + PII 脱敏（数字边界正则） |
| | `agents/support_agent.py` | **核心编排** | supervisor + 4 专家子图；Corrective RAG；真实 HITL |
| | `agents/tools.py` | 业务工具 | 知识库检索 / 订单查询（mock）/ 工单（mock） |
| **服务** | `service/service.py` | FastAPI | 流式 SSE / 多会话 / **中断恢复**（`Command(resume)`） |
| **记忆** | `memory/sqlite.py` | Checkpointer | SQLite（开发）→ Postgres（生产）；另有 LangGraph Store 长期记忆 |
| **评估** | `eval/golden.py` | 构建评测集 | grounded generation |
| | `eval/evaluate.py` / `metrics.py` | 检索评测 | Hit@k / MRR / nDCG@5 |
| | `eval/generate_eval.py` | 生成评测 | 引用准确率 / 幻觉率 / 转人工准确率 |
| | `eval/tune_weights.py` | **权重调参** | 80 验证/40 测试切分 + 网格搜索 |
| | `eval/mine_badcases.py` | 坏例回流 | gt 未进 top5 自动归档 |
| **可靠性** | `scripts/reliability_check.py` | 验收 | crash-resume + session isolation |
| **工程** | `docker/`, `compose.yaml` | 容器化 | 应用 + Qdrant + Postgres + Langfuse |

### Agent 编排结构（核心）

```
输入 → guard(注入检测) → route(5 类意图) ┬─ knowledge ─ SAQ多轮 → 检索 → generate(接地+流式) ─┬─ END
                                          │                                  └─[CANNOT_ANSWER]→ rewrite(限2) → 检索
                                          │                                  └─[CANNOT_ANSWER]&超限 → escalate
                                          ├─ order ── 实体抽取(正则+LLM) → 查单/反问
                                          ├─ complaint ─ 建工单 → interrupt() 挂起 ↘
                                          ├─ chitchat ─ 固定回复                    ├→ Checkpointer 持久化
                                          └─ escalate ─ 转人工 ↗                    ┘   人工 /resume 唤醒
```

---

## 三、全部指标（真实跑出）

### 3.1 检索（golden 120 题，文档级）

| 检索方式 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| Dense（bge-small-zh） | 0.542 | 0.858 | 0.908 | 0.712 | 0.758 |
| BM25-only（jieba） | 0.600 | 0.892 | 0.942 | 0.757 | 0.800 |
| Hybrid 等权 RRF（α=0.5） | 0.592 | 0.908 | 0.950 | 0.754 | 0.803 |
| **Hybrid 调参（α=0.1）** | **0.633** | **0.917** | **0.950** | **0.782** | **0.824** |

### 3.2 重排对比（同一批 30 题，公平对比）

| 方式 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| Hybrid 调参 | **0.567** | 0.833 | 0.900 | **0.715** | **0.759** |
| Hybrid + bge-reranker-v2-m3 | 0.500 ⬇️ | **0.867** ⬆️ | **0.933** ⬆️ | 0.692 ⬇️ | 0.750 ⬇️ |

### 3.3 权重调参（80 验证 / 40 held-out 测试）

| 数据集 | 配置 | Hit@1 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| 验证(80) | 等权 (0.5, 60) | 0.5125 | — | 0.6999 | — |
| 验证(80) | 调参 (0.1, 100) | **0.5625** | — | **0.7350** | — |
| **测试(40)** | 等权 (0.5, 60) | 0.7500 | 0.9750 | 0.8608 | 0.8887 |
| **测试(40)** | 调参 (0.1, 60) | **0.7750** | **1.0000** | **0.8800** | **0.9109** |

### 3.4 生成（15 题可回答 + 30 题负样本）

| 指标 | 数值 | 说明 |
|---|---|---|
| 引用准确率 | **100%**（36/36） | 回答中每个 URL 都来自检索到的文档，零编引用 |
| 转人工率（可回答问题） | 20%（3/15） | 自验证把不确定问题诚实拒答 |
| 幻觉率（实际交付回答中） | 16.7%（2/12） | 1 − faithfulness，仅统计真正答出去的回答 |
| 负样本转人工/闲聊准确率 | 83.3%（10/12） | 越界问题是否被正确拒答 |

### 3.5 工程与可靠性

| 项 | 结果 |
|---|---|
| 单元测试 | 7/7 通过 |
| crash-resume（中断跨进程恢复） | ✅ PASS |
| session isolation（多会话记忆隔离） | ✅ PASS |
| 检索延迟（修 localhost IPv6 后） | ~0.03s / 次 |
| 端到端延迟 | 闲聊 ~0.4s、订单 ~1s、知识问答 ~26s（**见短板 §5.1**） |

---

## 四、设计思路（关键决策 + 理由）

| 决策 | 选择 | 理由 |
|---|---|---|
| 编排框架 | **LangGraph** | 状态机精细控制；原生 interrupt/checkpoint/多 agent；CrewAI 黑盒、难做精细评估 |
| 向量库 | **Qdrant** | 单二进制、原生 hybrid、内存友好；ES 适合日志不适合主向量库，Milvus 单机更重 |
| 推理 | **API（DeepSeek）** | 质量高、评判可信、零部署；本地推理能力已在另一项目(ACE)证明，不重复 |
| 检索策略 | **hybrid + weighted RRF** | dense 擅语义、BM25 擅精确关键词；RRF 用名次回避量纲问题 |
| 权重 α=0.1 | **实验得出** | 80/40 切分网格搜索；与消融实验独立互证（都指向 BM25 更强） |
| **不用 rerank** | **实测决策** | 提升了 Hit@3/5 但**拉低 Hit@1/MRR**，加 CPU 成本 → 不采用（详见 §5.4） |
| Agent 架构 | **supervisor + 4 专家子图** | 各专家独立编译、可单独测试/替换；比"万能 ReAct"更省 token、更可控 |
| 防死循环 | MAX_REWRITES=2 | 业界 2-3 次，LangGraph 递归预算兜底 |
| 流式 vs 自验证冲突 | `skip_stream` 标签 + `[CANNOT_ANSWER]` 标记 | 内部节点不污染流；最终回答流式低 TTFT；严格前缀避免误杀 |
| 幻觉率定义 | **三次修正** | ① 上下文截断导致假 100% → ② 转人工被误判成幻觉 → ③ 转人工单列，只统计实际回答 |

---

## 五、短板模块（诚实列出，欢迎重点追问）

> 以下是我认为**最可能被评审挑战**的点，按严重度排序。每一条都如实标注，不掩饰。

### 5.1 【最严重】知识问答端到端延迟 ~26s

- **现象**：kb 路径 route + generate 两次 LLM 调用，DeepSeek 推理模型单次 ~6s+，叠加后 ~26s。
- **已做的优化**：把 grade/verify 合并进 generate（LLM 调用 4→2）。
- **未解决**：推理模型本身慢是根因。
- **建议方向**：换非推理模型（如 deepseek-chat）、加 Redis 缓存常见问题、流式 + 首字优先。

### 5.2 answer 评估样本过少（15 题）

- 引用准确率 100%、幻觉率 16.7% 都是 15 题样本，**统计意义不足**。
- **建议**：扩到 100+ 题；当前只应作为"方向性参考"。

### 5.3 负样本转人工准确率仅 83%

- 30 题负样本中约 5 个未正确拒答，安全敏感场景（订单/财务/法务）不够。
- 根因：意图分类用 LLM 单次判断，有模型波动。

### 5.4 重排未采用（但已实测）

- 这是**有数据支撑的决策**而非遗漏：rerank 在 30 题上 Hit@1 −6.7pp。
- 可能原因：bge-reranker-v2-m3 是通用模型、非本领域微调；30 题子集方差大。
- **建议**：若要用，需领域微调 reranker 或换更轻的模型，并在更大测试集上验证。

### 5.5 BM25 内存索引不可水平扩展

- 每次启动从 Qdrant 拉全量 chunk 到内存建 BM25；多 worker 会重复构建、难扩到百万级。
- **建议**：改用 Qdrant 原生 sparse vector，或持久化全文索引（SQLite FTS5 / Tantivy）。

### 5.6 意图分类用 LLM（慢且不稳）

- 每次都调一次 LLM 做意图分类，增加延迟与不确定性。
- **业界做法**：小模型（distilled BERT）做意图分类，固定输出空间、低延迟。

### 5.7 golden 集由 LLM 生成，无人工校验

- grounded generation 是冷启动标准做法，但**没有人工校对**，可能含坏题（已知空问题已加非空校验）。
- **建议**：抽样人工复核；线上流量接入后替换为真实问句。

### 5.8 无真实线上流量验证

- A/B 测试、压测、真实用户反馈闭环**均未做**（无流量），只在设计文档中描述。
- **建议**：这是作品集的天然边界，面试时应主动说明。

---

## 六、推荐改进（按投入产出比分级）

### P0（立刻可做，性价比最高）
1. **CI 评估门禁**：把 `eval.evaluate` 挂 GitHub Actions，指标低于阈值 fail —— 补齐"评估闭环"最后一块。
2. **延迟优化**：换非推理模型 + 常见问题缓存。
3. **answer 评估扩到 100 题**：让生成指标有统计意义。

### P1（中期，有明确收益）
4. 意图分类换小模型（降延迟、提稳定）
5. BM25 换 Qdrant 原生 sparse（可扩展）
6. Langfuse 实测接入 + trace 截图进 README
7. security benchmark（15~20 attack case + 误杀率）

### P2（长期/需真实流量）
8. A/B 测试框架、压测（k6/Locust）
9. 生产切 Postgres + 多副本高可用
10. bge-m3（dense+sparse 一体）+ 近重复去重

---

## 七、如何复现

```bash
# 依赖：conda env `agent` 已具备大部分
pip install langfuse langsmith langgraph-checkpoint-sqlite aiosqlite jieba

# 1. 语料（Qdrant 需运行在 6333）
git clone --depth 1 https://github.com/langgenius/dify-docs.git data/raw/dify-docs
python -m ingestion.loader
python -m retrieval.build_index

# 2. 评测（无需 LLM，纯检索）
python -m eval.evaluate --retriever dense|bm25|hybrid
python -m eval.tune_weights          # 权重调参实验

# 3. 生成评测（需 DEEPSEEK_API_KEY）
python -m eval.generate_eval 15

# 4. 可靠性验收
python scripts/reliability_check.py

# 5. 单测
pytest tests/ -q

# 6. 起服务
python src/run_service.py            # http://localhost:8080/docs
```

---

## 八、建议评审者重点关注

1. **检索调优的完整证据链**：dense 54.2% → BM25 60.0% → 等权 hybrid 59.2%（**失败**）→ 调参 hybrid 63.3%（**反超**）→ rerank 实测后弃用。这条链路上的**每一个结论都有数据**，包括两个反直觉的负面发现。
2. **Agent 编排的可控性**：为什么用 supervisor + 子图而不是万能 ReAct；为什么把"路由"和"生成"分离。
3. **真实 HITL**：`interrupt()` 挂起 + 跨进程 Checkpointer 恢复（已验证进程重启后仍能 resume）。
4. **评估的诚实性**：幻觉率定义的三次修正、转人工与幻觉的分离统计、小评测集高估性能的教训。
5. **已知短板的处理方式**：是否如实标注（§5），而非粉饰。
