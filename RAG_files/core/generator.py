from __future__ import annotations
import os
from typing import List, Dict, Generator

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

_model = None

SYSTEM_PROMPT = """You are an expert financial analyst assistant specializing in IFC (International Finance Corporation) annual reports.

Answer questions based ONLY on the provided context. If the information is not in the context, clearly state that you don't know.

Rules:
- Cite precise figures and always mention the fiscal year (FY24, FY23, etc.)
- For tables, present data in a structured way
- Indicate the source (page number, content type)
- Be concise and to the point
"""


def _get_model():
    global _model
    if _model is None:
        vertexai.init(
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION", "us-central1"),
        )
        _model = GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"))
    return _model


def generate_answer(
    query: str,
    context_docs: List[Dict],
    stream: bool = True,
) -> Generator[str, None, None]:
    model = _get_model()

    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        ctype = doc.get("content_type", "text")
        page = doc.get("page", "?")
        content = doc.get("content", "")
        context_parts.append(f"[Source {i} | Type: {ctype} | Page: {page}]\n{content}")

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context_str}

Question: {query}

Answer:"""

    config = GenerationConfig(temperature=0.1, max_output_tokens=1024)

    if stream:
        response = model.generate_content(prompt, generation_config=config, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    else:
        response = model.generate_content(prompt, generation_config=config)
        yield response.text


def caption_image(image_bytes: bytes, page: int) -> str:
    model = _get_model()
    image_part = Part.from_data(data=image_bytes, mime_type="image/png")
    prompt = (
        f"This image is from page {page} of the IFC Annual Report 2024. "
        "Describe in 2-3 sentences what this chart/graph/image shows: "
        "the type of visualization, key values, and what financial metric it represents."
    )
    response = model.generate_content([image_part, prompt])
    return response.text
