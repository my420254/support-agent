"""Embedding 封装：BAAI/bge-small-zh-v1.5（中文，512 维）。

baseline 先用本机缓存的 bge-small-zh；后续可升级 bge-m3（dense+sparse 一体）。
bge 系列查询端建议加前缀（见 _QUERY_PREFIX），默认关闭，作为后续优化项。
"""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    def __init__(self, model_name: str = _MODEL_NAME):
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
