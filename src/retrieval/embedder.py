"""Embedding provider：本地 bge-small-zh（默认）或云端 OpenAI 兼容 API。

为什么做可切换（工程权衡）：
- **本地**：离线可跑、零 API 成本；但打包 torch（~526MB）+ 模型（92MB），
  镜像 ~2GB，建索引在本机 CPU 上约 88s，部署到免费平台很吃力。
- **云端**：镜像可瘦到 ~200MB、建索引几十秒、免费平台能跑；
  但需要 key + 网络，内网部署不友好。

用 settings.EMBEDDING_PROVIDER 切换：local | remote。
注意：两种 provider 的向量维度不同（bge-small-zh=512，bge-m3=1024），切换后需重建索引。
"""

import os

# 离线加载：模型已本地缓存。不设置时，每次加载都会去 HF Hub 检查最新 commit，
# 在国内网络下 SSL 会卡住并重试，拖慢启动数十秒。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from core.settings import settings

_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # bge 系列查询端前缀


class LocalEmbedder:
    """本地 sentence-transformers 模型（默认 bge-small-zh，512 维）。"""

    def __init__(self, model_name: str = _LOCAL_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        return self.model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.model.encode(_QUERY_PREFIX + query, normalize_embeddings=True).tolist()


class RemoteEmbedder:
    """云端 OpenAI 兼容 embedding API（如硅基流动 SiliconFlow 托管的 BGE-M3）。

    收益：镜像瘦身（不打包 torch）、建索引快、可部署到免费平台。
    """

    def __init__(self):
        from openai import OpenAI

        if not settings.EMBEDDING_API_KEY or not settings.EMBEDDING_MODEL:
            raise ValueError("远端 embedding 需要配置 EMBEDDING_API_KEY 与 EMBEDDING_MODEL")
        self.client = OpenAI(
            base_url=settings.EMBEDDING_API_BASE, api_key=settings.EMBEDDING_API_KEY
        )
        self.model = settings.EMBEDDING_MODEL
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed_documents(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out

    def embed_query(self, query: str) -> list[float]:
        resp = self.client.embeddings.create(model=self.model, input=[query])
        return resp.data[0].embedding


def get_embedder() -> LocalEmbedder | RemoteEmbedder:
    """按配置返回 embedding 实现（local / remote）。"""
    if settings.EMBEDDING_PROVIDER == "remote":
        return RemoteEmbedder()
    return LocalEmbedder()
