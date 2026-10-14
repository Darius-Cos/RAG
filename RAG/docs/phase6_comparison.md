# Phase 6 — Multimodal Visual Pipeline Comparison

## Important disclaimer

The numbers in the table below are **illustrative, not measured**. They were not
produced by running RAGAS against both pipelines. A genuine comparison requires
running `evaluation/ragas_eval.py` with `phase="5"` and `phase="6"` and committing
the resulting CSVs. That step is pending and will replace this table.

---

## Implementation note: LLM-as-judge, not VLM embeddings

The spec asks for a ColPali / PaliGemma style retriever that produces patch-level
embeddings and ranks with MaxSim. What is actually implemented in
`core/colpali_retriever.py` is different:

| Spec requirement | What is implemented |
|---|---|
| VLM producing contextualised patch embeddings (ColPali, PaliGemma) | Gemini 2.0 Flash scores rendered page images as an LLM judge |
| Late-interaction MaxSim over patch tokens | Whole-page scoring; no patch segmentation |
| Separate embedding index for visual chunks | No visual index; candidate pages come from the text retriever |

**Why this substitute is reasonable here:**
Gemini 2.0 Flash is a strong multimodal model that can read charts, tables rendered
as images, and complex layouts that pure OCR-based chunking misses. Using it as a
visual judge captures most of the benefit of a VLM for this document without
requiring a separate embedding service.

**Where a real VLM embedder would be better:**
At scale (hundreds of documents, thousands of pages), LLM-as-judge scoring becomes
expensive and slow because it requires one Gemini call per candidate page. A true
ColPali embedder would pre-compute patch embeddings offline and run MaxSim at query
time in microseconds.

---

## Illustrative comparison (not from actual RAGAS runs)

| Metric | Phase 5 (caption-based text RAG) | Phase 6 (visual page scoring) |
|---|---|---|
| Faithfulness | ~0.84 | ~0.95 |
| Answer Relevancy | ~0.81 | ~0.92 |
| Context Precision | ~0.79 | ~0.89 |

These numbers reflect the expected direction of improvement, not measured values.
Phase 6 should score higher on faithfulness for questions answered by charts or
complex tables because Gemini reads the rendered image directly instead of relying
on extracted text or captions that may be incomplete.

---

## How to produce real numbers

```bash
# Run Phase 5 evaluation (standard pipeline)
python -c "
from core.rag_pipeline import RAGPipeline
from evaluation.ragas_eval import run_evaluation
p = RAGPipeline('resources/documents/ifc-annual-report-2024-financials.pdf')
p.load()
run_evaluation(p, phase='5')
"

# Run Phase 6 evaluation (visual pipeline)
python -c "
from core.rag_pipeline import RAGPipeline
from evaluation.ragas_eval import run_evaluation
p = RAGPipeline('resources/documents/ifc-annual-report-2024-financials.pdf')
p.load()
run_evaluation(p, phase='6', store='faiss')  # route queries through phase6 toggle in query()
"
```

Results land in `evaluation/results/phase5_results.csv` and
`evaluation/results/phase6_results.csv`.
