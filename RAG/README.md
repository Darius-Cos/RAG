# Module 2 — Retrieval-Augmented Generation (RAG) Capstone

> Grid University · Gen AI Training Program

This branch is the **target** of your submission Pull Request for Module 2. Branch off this branch, build your project, then open a PR back into `rag`. Two AI reviewers — Claude and Gemini — run in parallel on every PR and each posts a detailed sticky comment with the verdict, phase-by-phase analysis, and concrete action items. For the full submission flow, see the [`main` branch README](https://github.com/griddynamics/gridu-genai/blob/main/README.md).

> ⚠️ **Heads-up:** the Pull Request you will open targets this branch but **will never be merged.** The `rag` branch is an evaluation target only. Your professor reads your code and the AI reviews on the PR thread, then closes the PR. No code from any submission ever lands on `rag`.

---

## Overview

You will design, implement, and evaluate a sophisticated **Retrieval-Augmented Generation** system over the **IFC Annual Report 2024 — Financials** PDF. Start with text-only RAG, then progressively extend the system to handle tables, images (charts and graphs), and finally a fully multimodal end-to-end pipeline.

The dataset offers rich content:
- **Textual** — executive summaries, management discussions, policy text, footnotes.
- **Tabular** — financial statements, commitments, portfolio distributions.
- **Visual** — charts and graphs of financial trends and portfolio composition.

The PDF is provided in this branch at `resources/documents/ifc-annual-report-2024-financials.pdf`. The evaluation Q&A dataset lives at `resources/datasets/rag_evaluation_dataset.csv`.

---

## Technical requirements

| Area | Requirement |
|---|---|
| LLM | Gemini 2.0 Flash or newer. Use streaming, function calling, and JSON / structured output where appropriate. |
| SDK | Google GenAI SDK with **Vertex AI auth** through a GCP project — **no API keys**. |
| UI | Streamlit *or* Gradio. |
| Vector DB | **FAISS** *and* **Qdrant** (both required for the Phase 1 comparison). |
| PDF processing | Docling / PyMuPDF / Gemini multimodal capabilities. |
| RAG framework | **LangChain**. |
| Containerisation | Docker. |
| Observability | **Langfuse** (tracing wired into the running app). |
| Evaluation | **RAGAS** (against the provided Q&A dataset). |

---

## Data parsing requirements

All three sub-steps are mandatory:

1. **Text** — extract textual content + metadata (page number at minimum). *(Optional: detect document structure for advanced chunking.)*
2. **Images** — extract images; ask Gemini to generate descriptive captions; collect metadata. *(Optional: relate the image to surrounding text.)*
3. **Tables** — extract tables; save in a structured format (`.md`, `.html`, `.json`, `.csv`) or summarise. Collect metadata. *(Optional: relate the table to surrounding text.)*

---

## Phase 1 — Naive Text RAG

Build the baseline. Text only.

- Extract textual content from the PDF, clean it, and chunk it.
- Generate **dense embeddings** for the chunks.
- Build a **FAISS index** over the embeddings.
- Build a **Qdrant index** over the *same* embeddings (run Qdrant locally; `docker-compose` is fine).
- **Compare** FAISS vs. Qdrant — at minimum, retrieval quality on a few sample queries. Document the comparison.
- Implement a retriever that returns the top-k chunks for a query.
- Integrate the retrieved chunks with a Gemini model for answer generation.
- Build a Streamlit / Gradio UI for querying.
- Instrument the pipeline with **Langfuse** tracing.

### Sample query

```
User:   What is IFC's mission and how many member countries does it have?
System: IFC's mission is to end extreme poverty and boost shared prosperity
        on a livable planet. IFC is owned by 186 member countries.
        [retrieved context snippets]
```

---

## Phase 2 — RAG Pipeline Evaluation

Evaluate the system with **RAGAS** against the provided Q&A dataset.

- Use `resources/datasets/rag_evaluation_dataset.csv` (synthetically generated; you may extend it).
- Compute RAGAS metrics: context relevance, faithfulness, answer relevance, etc.
- *(Encouraged experimentation)* Try an LLM-as-Judge approach for nuanced or multimodal queries.

> Execute pipeline evaluation **after each subsequent phase** to track regressions.

---

## Phase 3 — Hybrid Search & Re-ranking

Improve retrieval relevance for textual context.

- **Re-ranking** — integrate a **cross-encoder** to re-rank the top-N results from the base retriever.
- **Metadata integration** — extract metadata (page number, section, content type) during preprocessing. Use it for filtering (e.g., "search only in pages 10–20").
- *(optional for interns)* **Sparse + hybrid retrieval** — implement a BM25 sparse retriever; combine dense and sparse scores into a hybrid pipeline.
- *(Optional)* Try a custom re-ranking approach (graph-based, LLM as expert).

### Sample query

```
User:   What was the Net Income for FY24 and FY23?
System: For FY24, IFC's Net Income was $1,485 million.
        For FY23, the Net Income was $672 million.
        [retrieved context snippets]
```

---

## Phase 4 — Advanced RAG Techniques *(all optional for interns)*

- **Semantic caching** — store and reuse answers for similar queries.
- **Multi-hop retrieval** — design a process where the LLM iteratively refines queries or combines information across multiple retrieval passes.

> Phase 4 sub-items are explicitly optional and will **not** lower your grade.

---

## Phase 5 — Multimodal RAG

### 5.1 — Incorporating Tables

- **Table indexing & retrieval** — make extracted tables retrievable (see *Data parsing requirements*).
- **Querying tabular data** — write prompts that let the LLM interpret table queries and synthesise answers (including simple calculations or direct extraction).
- **Integrated retrieval** — modify the retrieval pipeline to search across **both** text chunks and table representations.
- *(Optional)* Add plotting functionality, similar to Module 1.

### 5.2 — Incorporating Images (Charts & Graphs)

- **Image extraction & understanding** — extract chart/graph images from the PDF (see *Data parsing requirements*).
- **Querying visual data** — include image descriptions in the retrieval search space; write prompts that use the image-derived text to answer user queries.

### Sample multimodal query

```
User:   Show me the trend of IFC's Net Income from FY22 to FY24.
System: Based on Figure 1 (Income Measures), Net Income showed an upward trend:
        FY22 net loss of $464M → FY23 net income of $672M → FY24 net income of $1,485M.
        [retrieved context snippets + image description]
```

---

## Phase 6 — ColPali-like Multimodal RAG

Implement a more end-to-end multimodal RAG system using a **Vision-Language Model**.

- **Visual document ingestion** — convert document pages into images and segment them into patches.
- **Multimodal embedding** — integrate a VLM (e.g., **PaliGemma**, **ColPali**, **LLaVA**, GPT-4V) that produces contextualised embeddings from visual document patches.
- **Multimodal retrieval** — adapt query embedding and implement a **late-interaction** or **MaxSim** approach (à la ColBERT / ColPali) to rank patches against the query.
- **Generation with visual context** — feed the retrieved patches (or pages) to a multimodal LLM and let it synthesise answers from text + visuals.
- **Source attribution** — point back to the specific page or patch that contributed to the answer.
- **Comparison** — document how this pipeline compares to your previous (text + table + image-caption) pipeline.

---

## How to submit

1. Branch off `rag`: `git checkout rag && git checkout -b <user-ldap-id>/rag-submission`.
2. Build your project on that branch.
3. Open a Pull Request targeting the `rag` branch.
4. Two AI reviewers (Claude + Gemini) run automatically and each posts a sticky PR comment with verdict, technical-requirements table, per-phase analysis, and action items.
5. Push more commits to re-trigger the reviewers. Each bot updates its existing comment in place.
6. When you reach `passed` / `passed_with_notes` on both, request final review from your professor.

Good luck.
