"""建索引：corpus.jsonl → 分块 → 向量化 → 写入 Qdrant。"""

import json
import time
import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from ingestion.chunker import chunk_markdown
from retrieval.embedder import Embedder
from retrieval.qdrant_store import QdrantStore

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus.jsonl"


def build() -> None:
    docs = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line]

    chunks: list[dict] = []
    for doc in docs:
        for c in chunk_markdown(doc["content"]):
            # 嵌入文本 = 标题 + 小节标题 + 正文（标题有助于区分上下文）
            embed_text = c["text"]
            if c["heading"]:
                embed_text = c["heading"] + "\n" + c["text"]
            chunks.append({**doc, "heading": c["heading"], "chunk_text": c["text"], "embed_text": embed_text})

    print(f"{len(docs)} docs -> {len(chunks)} chunks")

    t0 = time.time()
    embedder = Embedder()
    vecs = embedder.embed_documents([c["embed_text"] for c in chunks])
    print(f"embedded {len(vecs)} chunks in {time.time() - t0:.1f}s (dim={embedder.dim})")

    store = QdrantStore()
    store.recreate(embedder.dim)
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=v,
            payload={
                "doc_id": c["id"],
                "title": c["title"],
                "section": c["section"],
                "url": c["url"],
                "heading": c["heading"],
                "text": c["chunk_text"],
            },
        )
        for c, v in zip(chunks, vecs)
    ]
    store.upsert(points)
    print(f"indexed {store.count()} points into collection '{store.collection}'")


if __name__ == "__main__":
    build()
