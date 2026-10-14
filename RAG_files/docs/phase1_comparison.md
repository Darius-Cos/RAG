# FAISS vs Qdrant Retrieval Quality Comparison

| Query | FAISS Latency | Qdrant Latency | Top-K Overlap |
|---|---|---|---|
| What is IFC mission? | 12ms | 45ms | 100% |
| What was IFC Net Income for FY24? | 15ms | 48ms | 100% |

### Findings
FAISS excels at local, low-latency in-memory lookup. Qdrant provides identical recall for dense vectors while offering superior production-grade metadata filtering capabilities.