# IFC RAG Assistant

RAG system for querying the IFC Annual Report 2024, built with Gemini, FAISS, Qdrant, Langfuse, and RAGAS.

## Project Structure

```
files/
├── app.py                      # Streamlit UI (4 pages)
├── docker-compose.yml          # Qdrant
├── requirements.txt
├── .env.example
├── data/pdfs/                  # Place the PDF here
├── core/
│   ├── pdf_parser.py           # Extracts text, tables, images from PDF
│   ├── embeddings.py           # Vertex AI text-embedding-004
│   ├── vector_store.py         # FAISS + Qdrant
│   ├── retriever.py            # Dense retrieval + cross-encoder re-ranking
│   ├── sparse_retriever.py     # BM25 + hybrid search
│   ├── semantic_cache.py       # Semantic cache (cosine similarity)
│   ├── generator.py            # Gemini 2.0 Flash streaming + image captioning
│   └── rag_pipeline.py         # Full orchestration + Langfuse tracing
├── evaluation/
│   └── ragas_eval.py           # RAGAS metrics
└── observability/
    └── langfuse_client.py      # Langfuse v4 client
```

## Setup

```bash
# 1. Configure environment
cp .env.example .env
# Fill in GCP_PROJECT_ID and Langfuse keys

# 2. Start Qdrant (optional)
docker compose up -d

# 3. Install dependencies
pip3 install -r requirements.txt

# 4. Add the PDF
cp ifc-annual-report-2024-financials.pdf data/pdfs/

# 5. Start the app
PYTHONPATH=. streamlit run app.py
```

## Pages

| Page | Description |
|------|-------------|
| Setup & Index | Upload PDF, build FAISS/Qdrant/BM25 index |
| Chat | Ask questions, filter by page/type, semantic cache |
| Evaluation | Run RAGAS metrics (faithfulness, relevancy) |
| Cache Settings | View and clear semantic cache |

## Sidebar Options

| Option | Description |
|--------|-------------|
| Qdrant | Enable Qdrant index alongside FAISS |
| Re-ranker | Cross-encoder for result re-ranking |
| Semantic Cache | Avoid duplicate Gemini calls |
| Hybrid BM25 | Combine dense + sparse retrieval |
| Index Tables | Extract and index PDF tables |
| Index Images | Gemini Vision captions charts/graphs |
| Top-K | Number of context chunks to retrieve |
| Vector Store | Which store to query (faiss/qdrant/both) |
