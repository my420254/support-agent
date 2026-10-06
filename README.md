# support-agent —— 企业知识库智能客服（Agentic RAG）

一个真实可上线的「企业知识库智能客服 Agent」，语料为 Dify 官方中文文档。核心能力：混合检索 + 纠正式 RAG + 工具调用 + 转人工 + 评估闭环 + 可观测。

> 设计思想、参数选择、调参路线见 [docs/DESIGN.md](docs/DESIGN.md)。

## 技术栈

LangGraph + FastAPI + Qdrant + DeepSeek API + bge-small-zh（embedding）+ SQLite（→Postgres）。

## 快速开始

```bash
# 1. 依赖（已有 conda env `agent`）
#    pip install langfuse langsmith langgraph-checkpoint-sqlite aiosqlite

# 2. 配置
cp .env.example .env   # 填 DEEPSEEK_API_KEY / DEEPSEEK_MODEL

# 3. 拉语料 + 建索引（Qdrant 需在 localhost:6333 运行）
python -m ingestion.loader          # data/processed/corpus.jsonl
python -m retrieval.build_index     # 分块→向量化→写入 Qdrant

# 4. 启动服务
python src/run_service.py
# 文档: http://localhost:8080/docs
```

## 目录结构

```
src/
  agents/          # 客服 agent（LangGraph 图 + 工具）
  core/            # 配置 settings + LLM 工厂
  ingestion/       # 语料清洗/分块（loader, cleaner, chunker）
  retrieval/       # 检索（embedder, qdrant_store, build_index）
  memory/          # checkpointer（SQLite → Postgres）
  schema/          # API 数据模型
  service/         # FastAPI 服务（流式/多会话/history）
data/              # 语料与产物（不提交 git）
docs/DESIGN.md     # 设计文档
```

## 里程碑

P0 脚手架 ✅ · P1 语料 ✅ · P2 检索基线 ✅ · P3 Agentic · P4 评估 · P5 上线
