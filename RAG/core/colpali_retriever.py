from __future__ import annotations
import fitz
from typing import List, Dict
from vertexai.generative_models import GenerativeModel, Part


# NOTE: This module does NOT implement true ColPali/VLM patch embeddings.
# Instead it uses Gemini as an LLM judge that scores and answers from rendered
# page images. The trade-off vs. a real VLM embedder (PaliGemma, ColPali):
#   - Simpler setup — no separate embedding model or MaxSim index needed.
#   - No patch-level late interaction; pages are scored whole, not by token/patch.
#   - Higher per-query cost (one Gemini call per candidate page for scoring).
# This is an acceptable substitute when the primary goal is visual grounding,
# but retrieval quality cannot match true late-interaction MaxSim ranking.


class MultimodalPageRetriever:
    """Renders PDF pages to PNG and uses a VLM to score and answer from them."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def render_page_to_bytes(self, page_number: int) -> bytes:
        page = self.doc[page_number - 1]
        pix = page.get_pixmap(dpi=150)
        return pix.tobytes("png")

    def visually_score_pages(self, query: str, candidate_pages: List[int]) -> List[int]:
        """
        Ask Gemini to rate each candidate page image against the query.
        Returns pages sorted by score descending.
        This is LLM-as-judge scoring, not patch embedding + MaxSim.
        """
        model = GenerativeModel("gemini-2.0-flash")
        scored_pages = []

        for page_num in candidate_pages[:3]:
            page_bytes = self.render_page_to_bytes(page_num)
            image_part = Part.from_data(data=page_bytes, mime_type="image/png")

            prompt = (
                f'Rate how relevant this page is to the query: "{query}". '
                "Reply with a single float between 0.0 (irrelevant) and 1.0 (highly relevant). "
                "Return only the number."
            )

            try:
                response = model.generate_content([image_part, prompt])
                score = float(response.text.strip())
            except Exception:
                score = 0.0

            scored_pages.append((page_num, score))

        scored_pages.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in scored_pages]


def run_phase6_pipeline(query: str, pdf_path: str, candidate_pages: List[int]) -> str:
    """Score candidate pages visually and answer from the best-scoring page."""
    retriever = MultimodalPageRetriever(pdf_path)
    best_pages = retriever.visually_score_pages(query, candidate_pages)

    if not best_pages:
        return "No relevant visual context found."

    top_page = best_pages[0]
    page_bytes = retriever.render_page_to_bytes(top_page)

    model = GenerativeModel("gemini-2.0-flash")
    image_part = Part.from_data(data=page_bytes, mime_type="image/png")

    prompt = (
        f"Answer the following question based ONLY on the provided page image.\n"
        f"Provide precise source attribution.\n\n"
        f"Question: {query}\n"
        f"Source Attribution: Page {top_page}"
    )

    response = model.generate_content([image_part, prompt])
    return response.text
