"""Qdrant 向量库封装：建集合、批量写入、检索。"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# 用 127.0.0.1 而非 localhost：Windows 上 localhost 先解析到 IPv6(::1)，
# 走慢路径导致检索 P99 高达 21s；127.0.0.1 走 IPv4 只要 ~50ms。
DEFAULT_URL = "http://127.0.0.1:6333"
DEFAULT_COLLECTION = "dify_docs"


class QdrantStore:
    def __init__(self, url: str = DEFAULT_URL, collection: str = DEFAULT_COLLECTION):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def recreate(self, dim: int) -> None:
        """删除并重建集合（幂等重建）。"""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def upsert(self, points: list[PointStruct], batch_size: int = 256) -> None:
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection, points=points[i : i + batch_size]
            )

    def count(self) -> int:
        return self.client.count(collection_name=self.collection).count

    def search(self, vector: list[float], limit: int = 5) -> list:
        res = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return res.points
