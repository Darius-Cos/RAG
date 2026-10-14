from __future__ import annotations
from typing import List, Dict, Optional

from core.embeddings import embed_query
from core.vector_store import FAISSStore, QdrantStore


class Retriever:
    def __init__(
        self,
        faiss: Optional[FAISSStore] = None,
        qdrant: Optional[QdrantStore] = None,
        use_reranker: bool = True,
    ):
        self.faiss = faiss
        self.qdrant = qdrant
        self._reranker = None

        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                print(f"Re-ranker unavailable: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        store: str = "faiss",
        content_types: Optional[List[str]] = None,
        page_range: Optional[tuple] = None,
    ) -> List[Dict]:
        query_emb = embed_query(query)
        results = []

        # Over-fetch more candidates when filters are active so that post-filter
        # results still fill top_k slots.
        fetch_k = top_k * 4 if (content_types or page_range) else top_k * 2

        if store == "faiss" and self.faiss:
            results.extend(self.faiss.search(query_emb, top_k=fetch_k))
        elif store == "qdrant" and self.qdrant:
            results.extend(self.qdrant.search(query_emb, top_k=fetch_k))

        if content_types:
            results = [r for r in results if r.get("content_type", "text") in content_types]

        if page_range:
            lo, hi = page_range
            results = [r for r in results if lo <= r.get("page", 0) <= hi]

        if self._reranker and results:
            pairs = [(query, r.get("content", "")) for r in results]
            scores = self._reranker.predict(pairs)
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)

        return results[:top_k]
