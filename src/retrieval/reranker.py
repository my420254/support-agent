"""Cross-encoder 重排：对检索候选做 (query, text) 语义重打分，提升 top-k 精度。

为什么需要重排：RRF 融合只按"排名倒数"加权，不真正理解语义；
cross-encoder 把 query 和候选文本拼一起进模型，做逐对语义打分，精度远高于 bi-encoder。
用在 hybrid 检索之后的候选精排（先召回 20 条，再重排取 top-k）。
"""

from sentence_transformers import CrossEncoder

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class Reranker:
    def __init__(self, model_name: str = _MODEL_NAME):
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query: str, docs: list[dict], top_k: int) -> list[dict]:
        """对候选 docs 打分并返回重排后的 top_k。"""
        if not docs:
            return []
        scores = self.model.predict([(query, d["text"]) for d in docs])
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        out = []
        for d, s in ranked[:top_k]:
            dd = dict(d)
            dd["score"] = round(float(s), 4)
            out.append(dd)
        return out
