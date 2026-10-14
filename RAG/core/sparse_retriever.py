from __future__ import annotations
import pickle
from pathlib import Path
from typing import List, Dict


class BM25Retriever:
    INDEX_PATH = Path("faiss_index/bm25.pkl")

    def __init__(self):
        self.bm25 = None
        self.documents: List[Dict] = []

    def build(self, documents: List[Dict]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            print("rank_bm25 not installed. Run: pip install rank-bm25")
            return

        self.documents = documents
        corpus = [doc.get("content", "").lower().split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        self.INDEX_PATH.parent.mkdir(exist_ok=True)
        with open(self.INDEX_PATH, "wb") as f:
            pickle.dump({"bm25": self.bm25, "documents": self.documents}, f)

    def load(self) -> bool:
        if not self.INDEX_PATH.exists():
            return False
        with open(self.INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.documents = data["documents"]
        return True

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.bm25:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = dict(self.documents[idx])
                doc["score"] = float(scores[idx])
                doc["store"] = "bm25"
                results.append(doc)
        return results

    def is_built(self) -> bool:
        return self.INDEX_PATH.exists()


def hybrid_search(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    alpha: float = 0.7,
    top_k: int = 5,
) -> List[Dict]:
    """
    Combine dense (FAISS/Qdrant) and sparse (BM25) results using stable keys.
    Fixed NameError by changing doc.get to r.get.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Dict] = {}

    if dense_results:
        max_d = max(r.get("score", 0) for r in dense_results) or 1
        for r in dense_results:
            # S-a schimbat doc.get in r.get pentru a repara NameError
            key = f"{r.get('page', 0)}_{r.get('chunk_index', 0)}_{r.get('content_type', 'text')}"
            scores[key] = scores.get(key, 0) + alpha * (r.get("score", 0) / max_d)
            docs[key] = r

    if sparse_results:
        max_s = max(r.get("score", 0) for r in sparse_results) or 1
        for r in sparse_results:
            # S-a schimbat doc.get in r.get pentru a repara NameError
            key = f"{r.get('page', 0)}_{r.get('chunk_index', 0)}_{r.get('content_type', 'text')}"
            scores[key] = scores.get(key, 0) + (1 - alpha) * (r.get("score", 0) / max_s)
            if key not in docs:
                docs[key] = r

    ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    results = []
    for key in ranked[:top_k]:
        doc = dict(docs[key])
        doc["score"] = scores[key]
        doc["store"] = "hybrid"
        results.append(doc)
    return results