# Phase 1 — FAISS vs Qdrant Retrieval Quality Comparison

## Methodology

Evaluation dataset: `resources/datasets/rag_evaluation_dataset.csv` (35 Q&A pairs covering text, table, image, and combination content types).  
Metric definitions:
- **Overlap@5** — fraction of the top-5 results shared between FAISS and Qdrant for the same query.
- **FAISS rank-1** / **Qdrant rank-1** — content_type of the highest-scored chunk returned.
- **Latency** — wall-clock time for a single query on a MacBook M2 (warm index, no reranker).

---

## Results across 10 representative queries

| # | Question (truncated) | FAISS lat | Qdrant lat | Overlap@5 | FAISS rank-1 type | Qdrant rank-1 type | Stores agree? |
|---|---|---|---|---|---|---|---|
| 1 | What is IFC's mission? | 11ms | 43ms | 5/5 | text | text | yes |
| 2 | What was IFC's Net Income for FY24? | 13ms | 47ms | 5/5 | table | table | yes |
| 3 | What was total assets as of June 30 2023? | 12ms | 46ms | 4/5 | table | table | yes |
| 4 | What is the stated mission of the IFC? | 11ms | 44ms | 5/5 | text | text | yes |
| 5 | What was Income Available for Designations FY24? | 14ms | 49ms | 3/5 | image | text | **no** |
| 6 | What were total disbursements excluding guarantees in FY24? | 13ms | 48ms | 3/5 | text | image | **no** |
| 7 | What was outstanding borrowings as of June 2022? | 15ms | 51ms | 4/5 | image | image | yes |
| 8 | What accounting principles does IFC follow? | 12ms | 45ms | 5/5 | text | text | yes |
| 9 | What were Charges on borrowings for FY23? | 13ms | 47ms | 5/5 | table | table | yes |
| 10 | How is Capital Utilization Ratio defined? | 12ms | 46ms | 4/5 | text | text | yes |

---

## Divergence analysis

Queries 5 and 6 show the most interesting difference. Both involve financial figures
that appear primarily in image-based charts (Figures 1 and 3 in the report). FAISS
ranked the image caption chunk first for query 5, while Qdrant promoted the
surrounding text chunk instead. Neither is wrong — the figure caption contains the
exact number but the text provides FY-over-FY context. In a hybrid retrieval setup
this divergence is useful: combining both stores increases the chance of returning
both the caption and the context in the top-5.

---

## Summary

| Metric | FAISS | Qdrant |
|---|---|---|
| Median latency | 12ms | 46ms |
| Mean Overlap@5 | 4.3/5 | — |
| Queries where stores disagree (rank-1 content_type differs) | — | 2/10 |

FAISS is roughly 4× faster for in-memory lookup. Qdrant matches FAISS on pure
retrieval quality for text/table queries and offers metadata filtering and
horizontal scaling that FAISS cannot provide. For a single-document use case FAISS
is sufficient; Qdrant becomes valuable if the corpus grows or if server-side payload
filters need to be applied before vector search.

---

## Known limitations

- Latency measured on a warm index; first-query overhead for FAISS (loading from
  disk) is ~800ms vs Qdrant's consistent ~50ms (runs as a service).
- Overlap@5 is a proxy for retrieval quality. A proper MRR or NDCG evaluation
  requires labelled relevance judgements per query, which the current dataset does
  not provide at the chunk level.
- Reranker was disabled for this comparison to isolate store-level differences.
