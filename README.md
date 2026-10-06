# Support Agent —— 生产级企业知识库智能客服（Agentic RAG）

一个**面向真实落地的企业客服 Agent**，语料为 Dify 官方中文文档（291 篇 / 98 万字符 / 4043 chunks）。核心不是"调个 RAG API"，而是一套**有评估闭环、有人机协同、有安全护栏、可观测、可容器化**的完整工程。

> 设计思想、参数选择、调优记录见 [docs/DESIGN.md](docs/DESIGN.md)。

## 架构

```
用户 → FastAPI 网关 → 安全护栏(注入检测+PII脱敏) → 意图路由(5类)
                                                        ↓
              ┌─────────────────────────────────────────┼──────────────────────┐
              ↓ kb                                      ↓ order                ↓ complaint/escalate
        知识专家(深)                                订单专家               工单/转人工
        SAQ多轮指代 → 混合检索                     实体抽取(正则+LLM)      interrupt() 挂起
        → Corrective RAG → 流式生成               +长期记忆(Store)        → Checkpointer 持久化
        (dense+BM25+RRF+重排)                      → 工具调用              → 人工 Resume 恢复
              └─────────────────────────────────────────┴──────────────────────┘
                                    ↑ 可观测(Langfuse) + 评估闭环(golden/坏例回流)
```

- **4 个专家子图**（独立编译、可单独替换/测试）：知识 / 订单 / 工单 / 闲聊
- **Corrective RAG**：检索→生成→`[CANNOT_ANSWER]`→改写重检索（限 2 次防死循环）
- **真实 HITL**：`interrupt()` 挂起 + Checkpointer 持久化 + 人工 `resume` 恢复（**跨进程重启也能恢复**）

## 核心指标（golden 集 120 题，真实跑出）

### 检索（文档级）

| 检索方式 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| Dense（bge-small-zh） | 0.542 | 0.858 | 0.908 | 0.712 | 0.758 |
| BM25（jieba） | 0.600 | 0.892 | 0.942 | 0.757 | 0.800 |
| Hybrid 等权 RRF（α=0.5, k=60） | 0.592 | 0.908 | 0.950 | 0.754 | 0.803 |
| **Hybrid 权重调参（α=0.1, k=60）** | **0.633** | **0.917** | **0.950** | **0.782** | **0.824** |

**权重调参（80 验证 / 40 测试切分，网格搜索 α×k）**：调参得 **dense 权重 α=0.1、BM25 权重 0.9**，与消融实验独立互证（都指向"BM25 在此语料更强"）。held-out 测试集 Hit@1 0.750→0.775、nDCG@5 0.889→0.911。

**重排对比（同一批 30 题）**：

| 方式 | Hit@1 | Hit@3 | Hit@5 | MRR | nDCG@5 |
|---|---|---|---|---|---|
| hybrid 调参（α=0.1） | **0.567** | 0.833 | 0.900 | **0.715** | **0.759** |
| hybrid + rerank | 0.500 ⬇️ | **0.867** ⬆️ | **0.933** ⬆️ | 0.692 ⬇️ | 0.750 ⬇️ |

**三条真实结论**：
1. **等权 RRF 的 hybrid（59.2%）打不过 BM25 单路（60.0%）**——诚实的负面发现；
2. **权重调参后 hybrid（63.3%）全面反超所有单路**——调参不是玄学，是有验证集/测试集的实验；
3. **重排提升深位召回但拉低首位精度**（Hit@1 −6.7pp），加 CPU 上 560M 参数的成本，**故最终不加重排**——有数据支撑的工程决策，不是"业界都上所以我也上"。

### 生成（15 题可回答 + 30 题负样本）

| 指标 | 数值 |
|---|---|
| 引用准确率（cited correctness） | **100%**（36/36） |
| 转人工率（可回答问题中的诚实拒答） | 20% |
| 幻觉率（实际交付回答中，1 - faithfulness） | 16.7% |
| 负样本转人工/闲聊准确率 | 83% |

## 真实工程踩坑（War Stories）

这些是**真实调试记录**，不是编的——每条都能展开讲：

1. **DeepSeek 推理模型不支持 `response_format` 和强制 `tool_choice`** → 意图/评分改用纯文本+解析
2. **Qdrant `localhost` 走 IPv6 慢路径** → 检索 P99 高达 21s，改 `127.0.0.1` 后 0.03s（700x）
3. **sentence-transformers 每次加载去 HF Hub 检查** → 国内 SSL 卡住拖慢启动，设 `HF_HUB_OFFLINE=1`
4. **评测把"转人工"误判成"幻觉"** → 转人工单列，幻觉率只在实际回答里统计
5. **LLM API 瞬断** → 退避重试（max_retries + 指数退避）
6. **PII 正则 `\b` 词边界对中文失效**（中文是 word 字符）→ 改数字边界 `(?<!\d)...(?!\d)`，且证件号先于手机号脱敏
7. **小测评集高估性能** → 68 题 hybrid Hit@1 一度 71%，扩到 120 题回落到 59%（evaluation leakage / sample-size sensitivity）
8. **bge-reranker-v2-m3 在 CPU 上极慢**（560M 参数）→ 生产需 GPU 或更轻的 reranker（cost/latency trade-off）
9. **等权 RRF 未必最优** → 80 验证/40 测试网格搜索权重，α=0.5→0.1 让 hybrid 从"打不过 BM25"变成"反超所有单路"（Hit@1 59.2%→63.3%）

## 快速开始

```bash
# 1. 依赖（conda env `agent` 已具备大部分）
pip install langfuse langsmith langgraph-checkpoint-sqlite aiosqlite jieba

# 2. 配置（.env 不提交，见 .env.example）
#    DEEPSEEK_API_KEY / DEEPSEEK_MODEL / QDRANT_URL

# 3. 拉语料 + 建索引（Qdrant 需运行在 6333）
python -m ingestion.loader
python -m retrieval.build_index

# 4. 启动服务
python src/run_service.py   # http://localhost:8080/docs

# 5. 评测
python -m eval.evaluate --retriever dense|bm25|hybrid|rerank
python -m eval.generate_eval 15        # 生成端指标
python scripts/reliability_check.py    # crash-resume + session isolation
```

## 目录结构

```
src/
  agents/        # supervisor + 4 专家子图 + 护栏
  core/          # 配置 + LLM 工厂（DeepSeek）
  ingestion/     # MDX 清洗 + 代码块感知分块
  retrieval/     # dense/BM25/hybrid/RRF/rerank
  memory/        # checkpointer（SQLite→Postgres）
  service/       # FastAPI 流式 + 中断恢复
  eval/          # golden/负样本/坏例挖掘/指标
scripts/         # reliability_check.py
docs/DESIGN.md   # 设计文档（架构+参数+调优+踩坑）
```

## 可靠性验收

```bash
python scripts/reliability_check.py
# ① crash-resume：投诉→interrupt 挂起→新进程重启→resume 恢复  ✅
# ② session isolation：用户A记住订单号，用户B查不到（记忆按 user_id 隔离）✅
```

## 技术栈

LangGraph + FastAPI + Qdrant + DeepSeek API + bge-small-zh(embedding) + bge-reranker-v2-m3(rerank) + SQLite/Postgres + Langfuse
