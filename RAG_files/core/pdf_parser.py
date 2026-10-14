from __future__ import annotations
import os
import io
from dataclasses import dataclass, field
from typing import Optional, Callable
from PIL import Image
import fitz  


@dataclass
class TextChunk:
    content: str
    page: int
    chunk_index: int
    content_type: str = "text"
    metadata: dict = field(default_factory=dict)


@dataclass
class TableChunk:
    content: str 
    page: int
    table_index: int
    content_type: str = "table"
    metadata: dict = field(default_factory=dict)


@dataclass
class ImageChunk:
    caption: str
    page: int
    image_index: int
    image_bytes: Optional[bytes] = None
    content_type: str = "image"
    metadata: dict = field(default_factory=dict)


class PDFParser:
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.doc = fitz.open(pdf_path)
        self.source = os.path.basename(pdf_path)

    def extract_text_chunks(self) -> list[TextChunk]:
        chunks = []
        chunk_index = 0

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text("text").strip()
            if not text:
                continue

            words = text.split()
            current: list[str] = []
            current_len = 0

            for word in words:
                current.append(word)
                current_len += len(word) + 1
                if current_len >= self.chunk_size:
                    chunks.append(TextChunk(
                        content=" ".join(current),
                        page=page_num + 1,
                        chunk_index=chunk_index,
                        metadata={"source": self.source, "page": page_num + 1},
                    ))
                    chunk_index += 1
                    overlap = current[-(self.chunk_overlap // 6):]
                    current = overlap
                    current_len = sum(len(w) + 1 for w in current)

            if current and current_len > 50:
                chunks.append(TextChunk(
                    content=" ".join(current),
                    page=page_num + 1,
                    chunk_index=chunk_index,
                    metadata={"source": self.source, "page": page_num + 1},
                ))
                chunk_index += 1

        return chunks

    def extract_tables(self) -> list[TableChunk]:
        tables = []
        table_index = 0

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            try:
                for table in page.find_tables():
                    data = table.extract()
                    if not data or len(data) < 2:
                        continue
                    headers = [str(h) if h else "" for h in data[0]]
                    md = "| " + " | ".join(headers) + " |\n"
                    md += "| " + " | ".join("---" for _ in headers) + " |\n"
                    for row in data[1:]:
                        md += "| " + " | ".join(str(c) if c else "" for c in row) + " |\n"

                    tables.append(TableChunk(
                        content=md,
                        page=page_num + 1,
                        table_index=table_index,
                        metadata={
                            "source": self.source,
                            "page": page_num + 1,
                            "rows": len(data),
                        },
                    ))
                    table_index += 1
            except Exception:
                continue

        return tables

    def extract_images(self, caption_fn: Optional[Callable] = None) -> list[ImageChunk]:
        images = []
        image_index = 0

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            for img_info in page.get_images(full=True):
                try:
                    xref = img_info[0]
                    base_image = self.doc.extract_image(xref)
                    img_bytes = base_image["image"]
                    img = Image.open(io.BytesIO(img_bytes))
                    if img.width < 100 or img.height < 100:
                        continue

                    caption = f"Image on page {page_num + 1}"
                    if caption_fn:
                        try:
                            caption = caption_fn(img_bytes, page_num + 1)
                        except Exception:
                            pass

                    images.append(ImageChunk(
                        caption=caption,
                        page=page_num + 1,
                        image_index=image_index,
                        image_bytes=img_bytes,
                        metadata={
                            "source": self.source,
                            "page": page_num + 1,
                            "width": img.width,
                            "height": img.height,
                        },
                    ))
                    image_index += 1
                except Exception:
                    continue

        return images

    def close(self):
        self.doc.close()
