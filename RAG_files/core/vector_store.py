from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np


class FAISSStore:
    INDEX_DIR = Path("faiss_index")

    def __init__(self):
        self.index = None
        self.documents: List[Dict] = []

    def build(self, embeddings: List[List[float]], documents: List[Dict]) -> None:
        import faiss
        self.INDEX_DIR.mkdir(exist_ok=True)
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = documents
        faiss.write_index(self.index, str(self.INDEX_DIR / "index.faiss"))
        with open(self.INDEX_DIR / "docs.pkl", "wb") as f:
            pickle.dump(documents, f)

    def load(self) -> bool:
        import faiss
        idx = self.INDEX_DIR / "index.faiss"
        docs = self.INDEX_DIR / "docs.pkl"
        if not idx.exists() or not docs.exists():
            return False
        self.index = faiss.read_index(str(idx))
        with open(docs, "rb") as f:
            self.documents = pickle.load(f)
        return True

    def search(self, query_emb: List[float], top_k: int = 5) -> List[Dict]:
        import faiss
        vec = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(vec)
        scores, idxs = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx >= 0:
                doc = dict(self.documents[idx])
                doc["score"] = float(score)
                doc["store"] = "faiss"
                results.append(doc)
        return results

    def is_built(self) -> bool:
        return (self.INDEX_DIR / "index.faiss").exists()


class QdrantStore:
    COLLECTION = "rag_ifc"

    def __init__(self, host: str = "localhost", port: int = 6333):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=host, port=port)

    def build(self, embeddings: List[List[float]], documents: List[Dict]) -> None:
        from qdrant_client.models import Distance, VectorParams, PointStruct
        dim = len(embeddings[0])
        self.client.recreate_collection(
            collection_name=self.COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=emb, payload=doc)
            for i, (emb, doc) in enumerate(zip(embeddings, documents))
        ]
        self.client.upsert(collection_name=self.COLLECTION, points=points)

    def search(self, query_emb: List[float], top_k: int = 5) -> List[Dict]:
        results = self.client.search(
            collection_name=self.COLLECTION,
            query_vector=query_emb,
            limit=top_k,
            with_payload=True,
        )
        docs = []
        for r in results:
            doc = dict(r.payload)
            doc["score"] = r.score
            doc["store"] = "qdrant"
            docs.append(doc)
        return docs

    def is_built(self) -> bool:
        try:
            info = self.client.get_collection(self.COLLECTION)
            return info.points_count > 0
        except Exception:
            return False
