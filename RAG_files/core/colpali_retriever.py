from __future__ import  annotations
import os
import fitz  
from typing import List, Dict
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Part

class MultimodalPageRetriever:
    """
    Phase 6 compliant multimodal retriever.
    Renders PDF pages to visual images and leverages visual understanding.
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    def render_page_to_bytes(self, page_number: int) -> bytes:
        """Renders a specific PDF page granularity to PNG bytes for VLM ingestion."""
        page = self.doc[page_number - 1]
        pix = page.get_pixmap(dpi=150)
        return pix.tobytes("png")

    def visually_score_pages(self, query: str, candidate_pages: List[int]) -> List[int]:
        """
        Simulates late-interaction/MaxSim patch ranking by forcing a VLM 
        to evaluate page images against the query semantics.
        """
        model = GenerativeModel("gemini-2.0-flash")
        scored_pages = []

        # Evaluam paginile candidat vizual
        for page_num in candidate_pages[:3]:  # Top 3 pagini cele mai probabile
            page_bytes = self.render_page_to_bytes(page_num)
            image_part = Part.from_data(data=page_bytes, mime_type="image/png")
            
            prompt = f"""Evaluate the relevance of this page image to the user query: "{query}".
            Respond with a single float score between 0.0 (irrelevant) and 1.0 (highly relevant).
            Do not include text, return only the number."""
            
            try:
                response = model.generate_content([image_part, prompt])
                score = float(response.text.strip())
            except:
                score = 0.0
                
            scored_pages.append((page_num, score))

        # Sortam descrescator dupa scorul MaxSim vizual
        scored_pages.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in scored_pages]

def run_phase6_pipeline(query: str, pdf_path: str, candidate_pages: List[int]) -> str:
    """Runs Phase 6 end-to-end multimodal generation using visual contexts."""
    retriever = MultimodalPageRetriever(pdf_path)
    best_pages = retriever.visually_score_pages(query, candidate_pages)
    
    if not best_pages:
        return "No relevant visual context found."
        
    # Luam cea mai buna pagina vizuala
    top_page = best_pages[0]
    page_bytes = retriever.render_page_to_bytes(top_page)
    
    model = GenerativeModel("gemini-2.0-flash")
    image_part = Part.from_data(data=page_bytes, mime_type="image/png")
    
    prompt = f"""Answer the following question based ONLY on the provided page image.
    Provide precise source attribution.
    
    Question: {query}
    Source Attribution: Page {top_page}"""
    
    response = model.generate_content([image_part, prompt])
    return response.text