from __future__ import annotations
import os
from typing import List

import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

_model = None


def _get_model():
    global _model
    if _model is None:
        vertexai.init(
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION", "us-central1"),
        )
        _model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    return _model

def embed_texts(texts: List[str], batch_size: int = 5) -> List[List[float]]:
    model = _get_model()
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = model.get_embeddings(batch)
        all_embeddings.extend([e.values for e in embeddings])
    return all_embeddings


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]
