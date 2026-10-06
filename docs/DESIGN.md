# 设计文档（DESIGN）

> 本文档解释**为什么这么设计**：总体思路、架构、参数选择、调参路线与对比表。
> 配合每个模块代码里的中文注释一起看。目标是"可控、可懂、不过度编码"。

## 1. 项目定位

一个**真实可上线**的「企业知识库智能客服 Agent（Agentic RAG）」。

- **语料**：Dify 官方中文文档（291 篇 / 98 万字符）+ 文档内真实 FAQ。
- **要证明的**：不是"能回答问题"，而是有真实指标、真实调试、可观测、可部署的工程能力。
- **刻意与已有 ACE 项目区分**：ACE 证明过本地推理 + 混合检索；本项目补的是生产级服务、真实评估闭环、可观测、可上线。

## 2. 总体架构

```
接入网关(FastAPI) → 会话/上下文(SQLite checkpointer) → 输入理解 → 检索 → Agent 编排 → 工具 → 生成交付
                                                         │        │         │         │
                                                     query改写   混合检索   路由/评分  知识库检索
                                                                +重排      纠正式改写  转人工
        ↑ 可观测 Langfuse（P5）贯穿：token/延迟/成本/每步 trace
        ↑ 评估闭环（P4）：golden 集 → 检索指标 + RAGAS → 坏例回流
```

当前实现进度对应架构：
- ✅ 已实现：服务层、checkpointer、检索（dense 基线）、工具（`lookup_knowledge_base`）
- ⏳ P3：Agent 编排（路由/评分/纠正式/转人工）
- ⏳ P4：评估闭环
- ⏳ P5：可观测 + 部署

## 3. 技术选型 + 理由（对比表）

| 决策 | 选择 | 备选 | 为什么选它 |
|---|---|---|---|
| Agent 编排 | **LangGraph** | CrewAI | 状态机式精细控制；原生支持流式/checkpoint/转人工(interrupt)/多 agent；简历已写、底座也是它。CrewAI 上手快但黑盒、难做精细评估 |
| 向量库 | **Qdrant** | Milvus / ES | 单二进制、原生支持 dense+sparse 混合检索(RRF)、内存友好（32G 机器足够）。ES 适合日志/分析，不适合当主向量库；Milvus 单机更重 |
| Embedding | **bge-small-zh-v1.5**(基线) → bge-m3 | OpenAI/API embedding | 基线用本机已缓存的模型（零下载、快）；bge-m3 支持 dense+sparse 一体，是混合检索标准，留作升级项 |
| 推理 | **DeepSeek API** | 本地 Ollama | API 质量高、RAGAS 评判才可信、零部署负担；本地推理能力 ACE 已证明，不重复 |
| 会话持久化 | **SQLite**(P0-P4) → Postgres(P5) | 仅内存 | 开发期零运维；生产切 Postgres（并发 + Docker compose） |
| 可观测 | **Langfuse**(P5) | Prometheus/OTel | LLM 语义 trace（每步 token/延迟/成本）是核心；Prometheus/OTel 是基础设施层、企业级再加 |
| 服务框架 | **FastAPI**（复用 agent-service-toolkit） | 自研 | 复用已验证的流式/多会话/checkpointer 工程骨架，避免重复踩坑 |

## 4. 数据管线

```
dify-docs 仓库(zho .mdx)  →  cleaner.py 清洗(frontmatter/JSX/图片/链接)
        →  loader.py 建元数据(标题/分区/原文URL)  →  corpus.jsonl
        →  chunker.py 标题感知分块  →  embedder.py 向量化  →  Qdrant
```

- **语料不提交进 git**（第三方内容 + 版权），只提交清洗/分块代码，`data/` 已 gitignore。
- **原文 URL 存进 payload**：生成回答时引用溯源用。

## 5. 当前参数 + 选择理由

| 参数 | 当前值 | 选择理由 |
|---|---|---|
| chunk_size | 400 字符 | < bge-small 512 token 上限，留标题拼接余量；再大会稀释检索精度 |
| overlap | 60 字符 | 防止切块把语义边界切断，代价是少量冗余 |
| 分块方式 | 标题感知（##/### 切 section，超长硬切） | 保留"文档 → 小节"上下文，引用更准 |
| 距离 | COSINE | embedding 已归一化，余弦=内积，标准做法 |
| top_k | 4 | 平衡上下文 token 成本与召回覆盖 |
| embedding 维度 | 512（bge-small-zh） | 模型固定 |
| 查询前缀 | bge 检索前缀 | bge 系列查询端加前缀提升召回（`embedder.py` 已写） |

## 6. 检索调优记录（golden 集 120 题，真实指标）

| 版本 | 检索方式 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---|---|---|---|---|---|
| v0 基线 | dense（bge-small-zh） | 0.5417 | 0.8583 | 0.9083 | 0.9417 | 0.7117 |
| 消融 | BM25-only（jieba） | 0.6000 | 0.8917 | 0.9417 | 0.9667 | 0.7566 |
| v1 | hybrid（dense+BM25+RRF） | 0.5917 | 0.9083 | 0.9500 | 0.9667 | 0.7535 |
| v2 | hybrid + bge-reranker 重排（30 题子集） | 0.500 | 0.867 | 0.933 | 0.933 | 0.693 |
| v3（计划） | bge-m3 + 近重复去重 | — | — | — | — | — |

**诚实的重排结论**（同 30 题对比 hybrid：Hit@1 0.500/Hit@3 0.833/Hit@5 0.900）：rerank 提升 Hit@3/Hit@5/nDCG（+3~4pp），但 **Hit@1 没变**，且 **BM25 单路 Hit@1 仍最强（0.567）**。加上 560M 参数在 CPU 上的高成本，结论是**当前语料规模下 rerank 增益有限、性价比不高**——线上可选择不加 rerank 省成本，这是实测后得出的权衡，而非堆技术名词。

**诚实的消融结论**：dense 最弱（54%）；BM25 在这个关键词密集的技术文档上已经很强（60%）；hybrid（RRF）≈ BM25（59%）。混合检索的主要价值是补 dense 的短板，但在关键词密集语料上并未显著超过纯 BM25——**下一个真正的提升杠杆是重排（cross-encoder 语义重打分）**。

**小测评集的教训**：68 题版 hybrid Hit@1 一度 71%，扩充到 120 题后回落到 59%——小测评集会高估性能，测评集必须够大、够覆盖（这也是为什么加了坏例回流机制）。

**性能踩坑（真实调试记录）**：
1. Qdrant 用 `localhost` 在 Windows 走 IPv6 慢路径，检索 P99 高达 21s → 改 `127.0.0.1` → ~0.03s；
2. sentence-transformers 每次加载去 HF Hub 检查 commit，国内 SSL 卡住重试拖慢启动 → 设 `HF_HUB_OFFLINE=1` 离线加载。
（修完后 68 题整轮评测从 ~6 分钟降到 ~7 秒。）

## 7. Agent 设计（P3）

```
意图路由 → 检索 → 相关性评分(grading) → [不够] 纠正式改写(限2次) → 工具
   → 带引用生成 → 自检(是否被文档支撑) → [不足] 转人工
```

- **防死循环**：max_iterations 上限（业界 5-6 次），LangGraph 递归预算兜底。
- **转人工（已实现真实 HITL）**：LangGraph `interrupt()` 挂起图执行 + Checkpointer 持久化状态快照 + 人工坐席通过同 thread 调 `/resume`（服务层 `Command(resume=...)`）唤醒，实现平滑人机混线。
- **输出护栏**：必须引用来源、查不到就明说并转人工，不编造（`support_agent.py` 的 INSTRUCTIONS 已体现，P3 落地成节点）。

**P3 踩坑记录（真实调试）**：DeepSeek v4.1-flash 是推理模型，不支持 `response_format`（JSON 模式）和强制 `tool_choice`，所以 `grade` 节点用「YES/NO 纯文本 + 解析」，不用 `with_structured_output`。

## 8. 评估体系（真实指标）

**golden 集**：120 题（grounded generation，按分区比例采样，来源可追溯）+ 30 题负样本（out-of-scope，测转人工）+ 坏例回流机制。

**检索指标**（文档级，120 题）：
| 检索方式 | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|
| dense（bge-small） | 0.542 | 0.858 | 0.908 | 0.712 |
| BM25-only | 0.600 | 0.892 | 0.942 | 0.757 |
| hybrid（RRF） | 0.592 | 0.908 | 0.950 | 0.754 |

**生成指标**（自验证版，15 题可回答 + 12 题负样本）：
| 指标 | 数值 |
|---|---|
| 转人工/闲聊准确率（负样本） | 83.3%（意图分类有模型波动） |
| 转人工率（可回答问题） | 20%（自验证把不确定问题诚实拒答） |
| 引用准确率 | 100% |
| 幻觉率（实际交付回答中） | 16.7% |

**踩坑记录（真实调试）**：① DeepSeek 推理模型不支持 response_format/tool_choice；② Qdrant localhost IPv6 慢路径（21s→0.03s）；③ HF Hub 离线；④ 幻觉率评测把"转人工"误判成"幻觉"（已修：转人工单列）；⑤ LLM API 瞬断需退避重试。

## 9. 已知局限与上线分级（诚实）

**上线分级**：
- 简历/GitHub 作品集：✅ 可收官（真实指标 + 完整闭环 + 可靠性与护栏）
- 内部 demo / 小规模 beta：✅ 可（需加监控 + 人工兜底）
- 真实客户大规模生产：❌ 暂不建议（以下局限未达标）

**主要局限**：
1. **answer 评估样本少**（15 题）——生成指标需扩到 100+ 才有统计意义；
2. **KB 端到端延迟 ~26s**（DeepSeek 推理模型本身慢 + route/generate 多次调用）——需换更快的非推理模型、流式、缓存常见问题；
3. **负样本转人工准确率 83%**（30 题中约 5 个未正确拒答）——安全敏感场景不够；
4. **rerank 在 CPU 上 560M 参数极慢**——生产需 GPU 或 ONNX INT8 量化；
5. golden 集是 grounded generation（无人工标准答案），无法做 answer correctness，只能做 faithfulness/abstention/引用正确率。

**迭代机制**：120 题检索回归 + 生成评测设为固定 release gate，每次改 retriever/prompt 自动跑一次，防止静默回退。

## 10. 里程碑状态

- [x] P0 脚手架 + 服务
- [x] P1 语料管线
- [x] P2 检索基线（dense + Qdrant）
- [x] P3 Agentic 编排（Corrective RAG + 转人工 + 防循环）
- [~] P4 评估 harness + 真实调优（检索指标✅ dense vs hybrid；RAGAS 生成指标待做）
- [ ] P5 可观测 + 前端 + Postgres + Docker 上线
