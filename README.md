# IFC Financial Report Assistant | Multimodal RAG

A **Retrieval-Augmented Generation (RAG)** application for exploring financial information in the **IFC Annual Report 2024**. The project combines **Gemini on Vertex AI, FAISS, Qdrant, BM25, cross-encoder re-ranking, RAGAS, and Streamlit** to answer questions using evidence retrieved from PDF documents.

> **Core question:** How can an AI assistant answer questions about a long financial PDF, retrieve evidence from different content types, and show where its answers came from?

## Overview

Instead of sending an entire PDF directly to a language model, the application extracts and indexes relevant material, retrieves the best-matching document chunks for a question, and passes that context to Gemini for answer generation.

The implementation explores **text, tables, and images** in financial documents, with optional hybrid retrieval, semantic caching, evaluation, and a visual-document retrieval experiment.

## Key Features

- Extract textual PDF content and attach page metadata.
- Identify tables and create searchable text representations.
- Generate descriptions of document images using a multimodal model.
- Build a local FAISS index and optionally a Qdrant collection.
- Combine dense retrieval with optional BM25 sparse search.
- Apply cross-encoder re-ranking to improve candidate ordering.
- Filter results by document page range and content type.
- Generate answers with Gemini and show supporting source snippets.
- Cache semantically similar questions to reduce repeat work.
- Run evaluation experiments with RAGAS.
- Use optional Langfuse tracing and LangChain mode.
- Explore a ColPali-like visual page retrieval pipeline.

## How It Works

```text
PDF document
    |
    v
PDF parsing (text / tables / images)
    |
    v
Chunking + metadata (page, type)
    |
    v
Vertex AI embeddings
    |
    +--------------------------+
    |                          |
    v                          v
FAISS / Qdrant             BM25 (optional)
    |                          |
    +-----------+--------------+
                |
                v
        Retrieval + Re-ranking
                |
                v
         Gemini + Context
                |
                v
     Answer + Source References
                |
                v
        RAGAS / Langfuse
```

## Repository Structure

```text
RAG/                              # GitHub repository root
├── README.md                     # Portfolio overview (this file)
├── RAG_files/                    # Self-contained application copy
│   ├── app.py                    # Streamlit UI
│   ├── core/
│   │   ├── pdf_parser.py         # PDF text, table and image extraction
│   │   ├── embeddings.py         # Vertex AI embeddings
│   │   ├── vector_store.py       # FAISS and Qdrant
│   │   ├── retriever.py          # Dense search + re-ranking
│   │   ├── sparse_retriever.py   # BM25 and hybrid retrieval
│   │   ├── semantic_cache.py     # Cache of previous answers
│   │   ├── generator.py          # Gemini-based answer generation
│   │   ├── langchain_rag.py      # LangChain implementation
│   │   ├── colpali_retriever.py  # Visual-document experiment
│   │   └── rag_pipeline.py       # Pipeline orchestration
│   ├── evaluation/ragas_eval.py  # Evaluation logic
│   ├── observability/            # Langfuse integration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── RAG/                            # Second training/project copy
    ├── app.py
    ├── core/
    ├── docs/                      # Retrieval comparison notes
    ├── evaluation/
    ├── observability/
    └── resources/datasets/         # Evaluation questions
```

**Repository note:** both `RAG/` and `RAG_files/` are currently checked in. The commands below use `RAG_files/`, which contains a concise application README. These two trees should eventually be consolidated to avoid duplicate maintenance.

## Core Components

### 1. PDF ingestion

`core/pdf_parser.py` extracts document material. Text is chunked, while tables and charts are represented in forms that can participate in retrieval. Indexed items retain information such as page number and content type.

### 2. Embeddings and vector databases

`core/embeddings.py` uses Vertex AI's `text-embedding-005` model. The system indexes the generated vectors in **FAISS**, and can also use **Qdrant** for retrieval.

### 3. Retrieval quality

A dense retriever finds semantically related content. Optional **BM25** adds lexical matching; **cross-encoder re-ranking** helps reorder candidate passages. The application exposes several of these options through the sidebar.

### 4. Grounded answers

`core/generator.py` supplies retrieved passages to Gemini and instructs the model to answer using the available evidence, including financial-year and page references where possible.

### 5. Evaluation and observability

`evaluation/ragas_eval.py` contains RAG evaluation logic; `observability/langfuse_client.py` enables optional tracing. The included evaluation dataset lives under the `RAG/` copy, so paths may need adjustment if you use `RAG_files/` as the sole app directory.

## Technology Stack

| Category | Technology | Role |
|---|---|---|
| Application | Python + Streamlit | Interactive document Q&A |
| LLM | Gemini on Vertex AI | Context-grounded generation and image captions |
| Embeddings | Vertex AI text-embedding-005 | Semantic vectors |
| Vector search | FAISS + Qdrant | Dense document retrieval |
| Sparse search | BM25 | Keyword relevance |
| Re-ranking | Sentence Transformers | Candidate ranking |
| PDF parsing | PyMuPDF | Text/document extraction |
| RAG framework | LangChain | Alternative orchestration mode |
| Evaluation | RAGAS | RAG-specific quality analysis |
| Tracing | Langfuse | Optional observability |
| Infrastructure | Docker Compose | Local service configuration |

## Getting Started

### Prerequisites

- Python, pip, and access to a personal Google Cloud project with Vertex AI enabled
- Google Cloud CLI for Application Default Credentials
- The IFC Annual Report 2024 PDF, obtained from an authorized source
- Docker Desktop if Qdrant is enabled

### 1. Clone and enter the application directory

```bash
git clone https://github.com/Darius-Cos/RAG.git
cd RAG/RAG_files
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure environment

Create a local `.env` file (do not commit it):

```dotenv
GCP_PROJECT_ID=your-own-gcp-project
GCP_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash
PDF_PATH=data/pdfs/ifc-annual-report-2024-financials.pdf
QDRANT_HOST=localhost
QDRANT_PORT=6333
# Optional observability:
# LANGFUSE_PUBLIC_KEY=...
# LANGFUSE_SECRET_KEY=...
```

Authenticate with `gcloud auth application-default login`.

### 3. Add the PDF

Use the **Setup & Index** page to upload the PDF, or place it in `RAG_files/data/pdfs/` under the configured filename. The document itself is not included in this repository.

### 4. Run the application

```bash
streamlit run app.py
```

FAISS can be used without Qdrant. If you enable Qdrant, start a compatible Qdrant service separately and make sure its connection settings match `.env`. The included Docker Compose setup is a starting point and may need local adjustments.

## Application Pages

| Page | Purpose |
|---|---|
| **Setup & Index** | Upload PDF and build/load indexes |
| **Chat** | Ask questions, configure retrieval, inspect sources |
| **Evaluation** | Run RAG quality checks |
| **Cache Settings** | View or clear cached results |

## Example Questions

- What is IFC's mission?
- How did IFC net income change between FY23 and FY24?
- Which financial metrics are shown in a particular chart?
- Find relevant information on a specified range of pages.

## Concepts Demonstrated

- Dense vs. sparse information retrieval and hybrid search.
- Chunking strategies, metadata filtering, and source attribution.
- LLM grounding and hallucination reduction through retrieved evidence.
- Table/image understanding for multimodal document workflows.
- RAGAS-based evaluation and observability.
- Semantic caching and retrieval experiments.

## Limitations and Future Improvements

- The IFC PDF is not included; indexing requires an authorized copy.
- Models, cloud credentials, and optionally Qdrant must be configured before running.
- The two application directories should be consolidated, and resource paths rechecked.
- Evaluate answer faithfulness, retrieval quality, and image understanding before quoting performance claims.
- Page citations and generated financial figures should be checked against the source report.

## Project Context

Developed from a technical training RAG exercise. Respect the original course materials' redistribution permissions and third-party document licensing.

## Author

**Darius Costache** · [GitHub](https://github.com/Darius-Cos)
