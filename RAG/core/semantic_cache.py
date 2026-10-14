from __future__ import annotations
import time
from typing import Optional
import numpy as np
from core.embeddings import embed_query


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.92, max_size: int = 100, ttl_seconds: int = 3600):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: list[dict] = []  # {embedding, answer, docs, query, timestamp}

    def _cosine_sim(self, a: list, b: list) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def get(self, query: str) -> Optional[dict]:
        now = time.time()
        query_emb = embed_query(query)
        for entry in self._cache:
            if now - entry["timestamp"] > self.ttl:
                continue
            sim = self._cosine_sim(query_emb, entry["embedding"])
            if sim >= self.threshold:
                return {"answer": entry["answer"], "docs": entry["docs"], "cached": True, "similarity": sim}
        return None

    def set(self, query: str, answer: str, docs: list) -> None:
        if len(self._cache) >= self.max_size:
            self._cache.pop(0)
        self._cache.append({
            "embedding": embed_query(query),
            "answer": answer,
            "docs": docs,
            "query": query,
            "timestamp": time.time(),
        })

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)
