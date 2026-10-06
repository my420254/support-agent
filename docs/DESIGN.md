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

## 6. 检索优化路线（真实调参记录会填在这里）

这是本项目的"真实调试"证据链，每一步都配指标（P4 用 golden 集量化）：

| 版本 | 检索方式 | 现状 | 观察到的真实问题 |
|---|---|---|---|
| v0 基线 | dense（bge-small-zh） | ✅ 已跑 | "如何部署 Dify"命中教程页而非部署文档；cloud/self-host 近重复文档互相挤排名 |
| v1（计划） | + BM25 sparse 混合（RRF） | ⏳ | 预期：关键词类问题（部署/报错）精度提升 |
| v2（计划） | + bge-reranker 重排 | ⏳ | 预期：top-k 精度再提升 |
| v3（计划） | bge-m3 + 近重复去重 | ⏳ | 预期：中英文+sparse 一体、去掉 cloud/self-host 重复 |

## 7. Agent 设计（P3）

```
意图路由 → 检索 → 相关性评分(grading) → [不够] 纠正式改写(限2次) → 工具
   → 带引用生成 → 自检(是否被文档支撑) → [不足] 转人工
```

- **防死循环**：max_iterations 上限（业界 5-6 次），LangGraph 递归预算兜底。
- **转人工**：LangGraph `interrupt()`，服务层已支持中断恢复（复用底座）。
- **输出护栏**：必须引用来源、查不到就明说并转人工，不编造（`support_agent.py` 的 INSTRUCTIONS 已体现，P3 落地成节点）。

**P3 踩坑记录（真实调试）**：DeepSeek v4.1-flash 是推理模型，不支持 `response_format`（JSON 模式）和强制 `tool_choice`，所以 `grade` 节点用「YES/NO 纯文本 + 解析」，不用 `with_structured_output`。

## 8. 评估设计（P4）

- **golden 集**：文档 FAQ 抽取 + 基于真实文档的 grounded 生成（来源标注）。
- **检索指标**：Recall@k / MRR / nDCG（分检索与生成两段评，否则分不清改检索还是改 prompt）。
- **生成指标**：RAGAS（faithfulness / answer relevancy / context precision & recall）。
- **客服特有**：幻觉率、转人工准确率、引用准确率。
- **性能**：首字延迟 / 总延迟 / token / 成本。
- **CI 门禁**：指标低于阈值 fail；坏例回流 golden 集。

## 9. 里程碑状态

- [x] P0 脚手架 + 服务
- [x] P1 语料管线
- [x] P2 检索基线（dense + Qdrant）
- [x] P3 Agentic 编排（Corrective RAG + 转人工 + 防循环）
- [ ] P4 评估 harness + 真实调优
- [ ] P5 可观测 + 前端 + Postgres + Docker 上线
