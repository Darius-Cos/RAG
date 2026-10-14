import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.set_page_config(page_title="IFC RAG Assistant", layout="wide")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

PDF_PATH = os.environ.get("PDF_PATH", "data/pdfs/ifc-annual-report-2024-financials.pdf")

with st.sidebar:
    st.title("IFC RAG Assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Setup & Index", "Chat", "Evaluation", "Cache Settings"],
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Configuration")

    use_qdrant   = st.toggle("Qdrant", value=False)
    use_reranker = st.toggle("Re-ranker", value=True)
    use_cache    = st.toggle("Semantic Cache", value=True)
    use_hybrid   = st.toggle("Hybrid BM25", value=False)
    index_tables = st.toggle("Index Tables", value=True)
    index_images = st.toggle("Index Images", value=True)   # default True per reviewer
    use_langchain = st.toggle("LangChain Mode", value=False)  # NEW: LangChain toggle
    top_k        = st.slider("Top-K", 3, 10, 5)
    store        = st.selectbox("Vector Store", ["faiss", "qdrant", "both"])

    st.divider()
    if st.session_state.indexed:
        st.success("Index ready")
        if use_langchain:
            st.info("LangChain LCEL chain active")
        if st.session_state.pipeline and st.session_state.pipeline.cache:
            st.caption(f"Cache: {st.session_state.pipeline.cache.size()} entries")
    else:
        st.warning("Not indexed yet")


# ── Setup & Index ──────────────────────────────────────────────────────────────
if page == "Setup & Index":
    st.header("Setup & Index")
    st.markdown("Upload the IFC Annual Report 2024 PDF and build the vector index.")

    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        os.makedirs("data/pdfs", exist_ok=True)
        with open(PDF_PATH, "wb") as f:
            f.write(uploaded.read())
        st.success(f"PDF saved: `{PDF_PATH}`")

    if os.path.exists(PDF_PATH):
        st.info(f"PDF found: `{PDF_PATH}`")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Build Index", type="primary"):
                from core.rag_pipeline import RAGPipeline

                pipeline = RAGPipeline(
                        pdf_path=PDF_PATH,
                        use_qdrant=use_qdrant,
                        use_reranker=use_reranker,  
                        use_cache=use_cache,
                        use_hybrid=use_hybrid,
                        index_tables=index_tables,
                        index_images=index_images,
                    )

                status = st.empty()

                def update(msg):
                    status.info(msg)

                with st.spinner("Indexing in progress"):
                    stats = pipeline.index(progress_callback=update)

                st.session_state.pipeline = pipeline
                st.session_state.indexed = True
                status.empty()
                st.success(
                    f"Indexed: **{stats['text']}** text chunks | "
                    f"**{stats['tables']}** tables | "
                    f"**{stats['images']}** images | "
                    f"Total: **{stats['total']}** chunks"
                )

        with col2:
            if not st.session_state.indexed and os.path.exists("faiss_index/index.faiss"):
                if st.button("Load Existing Index"):
                    from core.rag_pipeline import RAGPipeline
                    pipeline = RAGPipeline(
                        pdf_path=PDF_PATH,
                        use_qdrant=use_qdrant,
                        use_reranker=use_reranker,  # Adaugat obligatoriu
                        use_cache=use_cache,
                        use_hybrid=use_hybrid,
                        index_tables=index_tables,  # Adaugat obligatoriu
                        index_images=index_images,  # Adaugat obligatoriu
                    )
                    pipeline.load()
                    st.session_state.pipeline = pipeline
                    st.session_state.indexed = True
                    st.success("Index loaded!")
                    st.rerun()
    else:
        st.warning("PDF not found. Please upload it above.")

    st.divider()
    st.subheader("How indexing works")
    st.markdown("""
    1. **Text extraction** — PyMuPDF splits text into ~1000 character chunks with overlap
    2. **Table extraction** — Detects tables and converts them to Markdown format
    3. **Image extraction** *(optional)* — Gemini Vision generates descriptions for charts
    4. **Embeddings** — text-embedding-005 (Vertex AI) vectorizes each chunk
    5. **FAISS** — builds a local vector index (fast, offline)
    6. **Qdrant** *(optional)* — builds index in Qdrant service (advanced filtering)
    7. **BM25** *(optional)* — sparse index for hybrid search
    """)


# ── Chat ───────────────────────────────────────────────────────────────────────
elif page == "Chat":
    st.header("Chat with IFC Annual Report 2024")

    if not st.session_state.indexed or st.session_state.pipeline is None:
        st.warning("Go to **Setup & Index** and build the index first.")
        st.stop()

    # Show active mode badge
    if use_langchain:
        st.info("🦜 **LangChain Mode** — using LCEL chain with structured JSON output")
    else:
        st.info("⚡ **Native Mode** — using direct RAG pipeline")

    col_chat, col_filters = st.columns([3, 1])

    with col_filters:
        st.subheader("Filters")
        content_filter = st.multiselect(
            "Content type",
            ["text", "table", "image"],
            default=["text", "table", "image"],
        )
        use_page_filter = st.checkbox("Filter by page range")
        if use_page_filter:
            page_min = st.number_input("From page", min_value=1, value=1)
            page_max = st.number_input("To page", min_value=1, value=300)
        else:
            page_min, page_max = 1, 9999

        st.divider()
        st.subheader("Example questions")
        examples = [
            "What is IFC's mission?",
            "What was IFC Net Income for FY24?",
            "Which region had the highest portfolio in FY24?",
            "What was the percentage change in total commitments FY23 to FY24?",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex[:20]}"):
                st.session_state._example_query = ex
                st.rerun()

    with col_chat:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("from_cache"):
                    st.caption("⚡ From semantic cache")
                if msg.get("langchain_mode"):
                    st.caption("🦜 Answered via LangChain LCEL")
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"**Page {s.get('page', '?')}** | "
                                f"Type: `{s.get('content_type', 'text')}` | "
                                f"Score: {s.get('score', 0):.3f}"
                            )
                            st.caption(s.get("content", "")[:300] + "...")
                            st.divider()

        prefill = st.session_state.pop("_example_query", None)
        user_input = st.chat_input("Ask about the IFC Annual Report 2024...") or prefill

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                page_range = (page_min, page_max) if use_page_filter else None

                # ── ROUTING: LangChain vs Native ───────────────────────────────
                if use_langchain:
                    from core.langchain_rag import langchain_query
                    import json
                    
                    answer_gen, docs = langchain_query(
                        question=user_input,
                        faiss_store=st.session_state.pipeline.faiss_store,
                        top_k=top_k,
                        stream=False, # Recomandat stocarea intregului string pentru a parsa JSON curat fara caractere brute
                    )
                    from_cache = False
                    
                    raw_answer = "".join(answer_gen)
                    try:
                        json_res = json.loads(raw_answer)
                        full_answer = json_res.get("answer", raw_answer)
                        citations = json_res.get("citations", [])
                        if citations:
                            full_answer += f"\n\n**Citations (Pages):** {', '.join(map(str, citations))}"
                    except Exception:
                        full_answer = raw_answer
                    st.markdown(full_answer)
                else:
                    answer_gen, docs, from_cache = st.session_state.pipeline.query(
                        question=user_input,
                        top_k=top_k,
                        store=store,
                        content_types=content_filter or None,
                        page_range=page_range,
                        stream=True,
                    )
                    if from_cache:
                        st.caption("⚡ Answer from semantic cache")
                    full_answer = st.write_stream(answer_gen)

                with st.expander("Sources"):
                    for s in docs:
                        st.markdown(
                            f"**Page {s.get('page', '?')}** | "
                            f"Type: `{s.get('content_type', 'text')}` | "
                            f"Score: {s.get('score', 0):.3f}"
                        )
                        # Show image content indicator
                        if s.get("content_type") == "image":
                            st.caption("📷 Image chunk — caption below:")
                        st.caption(s.get("content", "")[:300] + "...")
                        st.divider()

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_answer,
                "sources": docs,
                "from_cache": from_cache,
                "langchain_mode": use_langchain,
            })

    if st.session_state.chat_history:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()


# ── Evaluation ─────────────────────────────────────────────────────────────────
elif page == "Evaluation":
    st.header("RAGAS Evaluation")

    if not st.session_state.indexed or st.session_state.pipeline is None:
        st.warning("Build the index first.")
        st.stop()

    st.markdown("""
    Evaluate the RAG pipeline quality using **RAGAS** — automated metrics:
    - **Faithfulness** — is the answer faithful to the retrieved context?
    - **Answer Relevancy** — is the answer relevant to the question?
    - **Context Precision** — are the retrieved chunks relevant to the question?
    - **Context Recall** — did we retrieve all necessary context?
    """)

    col1, col2 = st.columns(2)
    with col1:
        eval_store = st.selectbox("Store for evaluation", ["faiss", "qdrant"])
        eval_top_k = st.slider("Top-K for evaluation", 3, 10, 5)
    with col2:
        st.markdown("**Evaluation dataset:**")
        st.markdown("Q&A pairs from `resources/datasets/rag_evaluation_dataset.csv`")
        st.markdown("_(falls back to 5 hardcoded pairs if CSV not found)_")

    if st.button("Run Evaluation", type="primary"):
        from evaluation.ragas_eval import run_evaluation

        with st.spinner("Running RAGAS evaluation with Gemini judge..."):
            try:
                df = run_evaluation(
                    pipeline=st.session_state.pipeline,
                    top_k=eval_top_k,
                    store=eval_store,
                )
                st.success("Evaluation complete!")
                st.dataframe(df, use_container_width=True)

                numeric_cols = df.select_dtypes(include="number").columns.tolist()
                if numeric_cols:
                    st.subheader("Average metrics")
                    cols = st.columns(len(numeric_cols))
                    for col, metric in zip(cols, numeric_cols):
                        val = df[metric].mean()
                        col.metric(
                            metric,
                            f"{val:.3f}",
                            delta="good" if val > 0.7 else "needs improvement",
                        )
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.info("Make sure Vertex AI credentials are configured in .env")


# ── Cache Settings ─────────────────────────────────────────────────────────────
elif page == "Cache Settings":
    st.header("Semantic Cache")

    if not st.session_state.pipeline or not st.session_state.pipeline.cache:
        st.warning("Cache is not active. Enable 'Semantic Cache' in the sidebar and rebuild the pipeline.")
        st.stop()

    cache = st.session_state.pipeline.cache

    col1, col2, col3 = st.columns(3)
    col1.metric("Cache entries", cache.size())
    col2.metric("Similarity threshold", f"{cache.threshold:.2f}")
    col3.metric("TTL (seconds)", cache.ttl)

    st.divider()
    st.markdown("""
    **How does Semantic Cache work?**

    Before every Gemini call, the system:
    1. Generates an embedding for the current question
    2. Compares it with embeddings of previous questions in cache
    3. If cosine similarity > **0.92**, returns the cached answer instantly
    4. Otherwise, runs the full pipeline and saves the answer to cache

    Benefits: reduced latency, reduced API costs, instant answers for similar questions.
    """)

    if st.button("Clear cache", type="secondary"):
        cache.clear()
        st.success("Cache cleared!")
        st.rerun()