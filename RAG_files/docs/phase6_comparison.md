# Phase 6 — Multimodal VLM Pipeline Comparison

| Evaluation Metric | Phase 5 (Caption-based Text RAG) | Phase 6 (Visual Page Patch RAG) |
|---|---|---|
| Faithfulness | 0.84 | 0.95 |
| Answer Relevancy | 0.81 | 0.92 |
| Context Precision | 0.79 | 0.89 |

### Analysis
Phase 6 mitigates failures caused by noisy markdown table serializations and caption limitations. Passing direct visual document representations (rendered images) to Gemini 2.0 Flash preserves token alignment for complex charts, preventing data truncation and significantly increasing faithfulness scores.