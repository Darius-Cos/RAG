from __future__ import annotations
import os
from typing import List, Dict, Optional, Generator

from core.pdf_parser import PDFParser
from core.embeddings import embed_texts
from core.vector_store import FAISSStore, QdrantStore
from core.retriever import Retriever
from core.sparse_retriever import BM25Retriever, hybrid_search
from core.semantic_cache import SemanticCache
from core.generator import generate_answer, caption_image
from core.colpali_retriever import MultimodalPageRetriever, run_phase6_pipeline
from observability.langfuse_client import get_langfuse_client


class RAGPipeline:
    def __init__(
        self,
        pdf_path: str,
        use_qdrant: bool = True,
        use_reranker: bool = True,
        use_cache: bool = True,
        use_hybrid: bool = False,
        index_tables: bool = True,
        index_images: bool = True,
    ):
        self.pdf_path = pdf_path
        self.use_qdrant = use_qdrant
        self.use_reranker = use_reranker
        self.use_cache = use_cache
        self.use_hybrid = use_hybrid
        self.index_tables = index_tables
        self.index_images = index_images

        self.faiss = FAISSStore()
        self.qdrant = QdrantStore(
            host=os.environ.get("QDRANT_HOST", "localhost"),
            port=int(os.environ.get("QDRANT_PORT", 6333)),
        ) if use_qdrant else None
        self.bm25 = BM25Retriever() if use_hybrid else None
        self.cache = SemanticCache() if use_cache else None

        self.retriever: Optional[Retriever] = None
        self.langfuse = get_langfuse_client()
        self._all_docs: List[Dict] = []

    def is_indexed(self) -> bool:
        return self.faiss.is_built()

    def index(self, progress_callback=None) -> Dict:
        parser = PDFParser(self.pdf_path)

        if progress_callback:
            progress_callback("Extracting text...")
        text_chunks = parser.extract_text_chunks()

        table_chunks = []
        if self.index_tables:
            if progress_callback:
                progress_callback("Extracting tables...")
            table_chunks = parser.extract_tables()

        image_chunks = []
        if self.index_images:
            if progress_callback:
                progress_callback("Extracting and captioning images...")
            image_chunks = parser.extract_images(caption_fn=caption_image)

        parser.close()

        docs = []
        for idx, c in enumerate(text_chunks):
            docs.append({
                "content": c.content,
                "content_type": "text",
                "page": c.page,
                "chunk_index": idx,
                "metadata": c.metadata,
            })

        for idx, c in enumerate(table_chunks):
            docs.append({
                "content": c.content,
                "content_type": "table",
                "page": c.page,
                "chunk_index": idx,
                "metadata": c.metadata,
            })

        for idx, c in enumerate(image_chunks):
            docs.append({
                "content": c.caption,
                "content_type": "image",
                "page": c.page,
                "chunk_index": idx,
                "metadata": c.metadata,
            })

        self._all_docs = docs

        if progress_callback:
            progress_callback(f"Generating embeddings for {len(docs)} chunks...")
        embeddings = embed_texts([d["content"] for d in docs])

        if progress_callback:
            progress_callback("Building FAISS index...")
        self.faiss.build(embeddings, docs)

        if self.qdrant:
            if progress_callback:
                progress_callback("Building Qdrant index...")
            self.qdrant.build(embeddings, docs)

        if self.bm25:
            if progress_callback:
                progress_callback("Building BM25 index...")
            self.bm25.build(docs)

        self._init_retriever()

        return {
            "text": len(text_chunks),
            "tables": len(table_chunks),
            "images": len(image_chunks),
            "total": len(docs),
        }

    def load(self):
        self.faiss.load()
        self._all_docs = self.faiss.documents
        # Qdrant connects via HTTP so no explicit load is needed;
        # BM25 index is serialized to disk and needs to be reloaded.
        if self.bm25:
            self.bm25.load()
        self._init_retriever()

    def _init_retriever(self):
        self.retriever = Retriever(
            faiss=self.faiss,
            qdrant=self.qdrant,
            use_reranker=self.use_reranker,
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        store: str = "faiss",
        content_types: Optional[List[str]] = None,
        page_range: Optional[tuple] = None,
        stream: bool = True,
        use_phase6_visual: bool = False,
    ) -> tuple:
        """
        Returns (answer_generator, context_docs, from_cache).
        Set use_phase6_visual=True to route through the visual page pipeline.
        """
        if not self.retriever:
            self.load()

        if self.cache:
            cached = self.cache.get(question)
            if cached:
                def _cached_gen():
                    yield cached["answer"]
                return _cached_gen(), cached["docs"], True

        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="rag-query",
            input={"question": question, "store": store},
        ) as span:

            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="retrieval",
            ) as ret:
                dense_docs = self.retriever.retrieve(
                    query=question,
                    top_k=top_k,
                    store=store,
                    content_types=content_types,
                    page_range=page_range,
                )

                if self.bm25 and self.bm25.is_built():
                    sparse_docs = self.bm25.search(question, top_k=top_k)
                    docs = hybrid_search(dense_docs, sparse_docs, alpha=0.7, top_k=top_k)
                else:
                    docs = dense_docs

                ret.update(output={"num_docs": len(docs)})

            # Phase 6 is triggered only via the explicit toggle, not by content type.
            if use_phase6_visual and docs:
                candidate_pages = list(set([d.get("page", 1) for d in docs]))
                phase6_answer = run_phase6_pipeline(question, self.pdf_path, candidate_pages)

                span.update(output={"context_docs": len(docs), "phase6_active": True})

            else:
                phase6_answer = None
                answer_gen = generate_answer(question, docs, stream=stream)
                span.update(output={"context_docs": len(docs)})

        self.langfuse.flush()

        if phase6_answer is not None:
            if self.cache:
                self.cache.set(question, phase6_answer, docs)
            def _phase6_gen():
                yield phase6_answer
            return _phase6_gen(), docs, False

        # Wrap the generator so the answer is cached after streaming completes.
        if self.cache:
            collected: List[str] = []

            def _caching_gen():
                for chunk in answer_gen:
                    collected.append(chunk)
                    yield chunk
                self.cache.set(question, "".join(collected), docs)

            return _caching_gen(), docs, False

        return answer_gen, docs, False
