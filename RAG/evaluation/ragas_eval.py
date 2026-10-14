from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

SAMPLE_QA = [
    {
        "question": "What is IFC's mission?",
        "ground_truth": "IFC's mission is to end extreme poverty and boost shared prosperity on a livable planet.",
    },
    {
        "question": "How many member countries does IFC have?",
        "ground_truth": "IFC is owned by 186 member countries.",
    },
    {
        "question": "What was IFC's Net Income for FY24?",
        "ground_truth": "IFC's Net Income for FY24 was $1,485 million.",
    },
    {
        "question": "What was IFC's Net Income for FY23?",
        "ground_truth": "IFC's Net Income for FY23 was $672 million.",
    },
    {
        "question": "Which region had the highest disbursed investment portfolio in FY24?",
        "ground_truth": "The Asia and Pacific region had the highest disbursed investment portfolio in FY24 with $18.9 billion.",
    },
]

CSV_PATH = Path("resources/datasets/rag_evaluation_dataset.csv")


def load_qa_pairs() -> List[Dict]:
    """Load Q&A pairs from CSV or fall back to SAMPLE_QA."""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        # col_map maps existing_col → canonical_col
        col_map = {}
        if "question" in df.columns:
            col_map["question"] = "question"
        elif "query" in df.columns:
            col_map["query"] = "question"
        if "ground_truth" in df.columns:
            col_map["ground_truth"] = "ground_truth"
        elif "answer" in df.columns:
            col_map["answer"] = "ground_truth"
        if len(col_map) >= 2:
            # Only rename columns that actually need renaming (source != target)
            rename_map = {k: v for k, v in col_map.items() if k != v}
            df = df.rename(columns=rename_map)
            qa_pairs = df[["question", "ground_truth"]].dropna().to_dict("records")
            print(f"[RAGAS] Loaded {len(qa_pairs)} Q&A pairs from {CSV_PATH}")
            return qa_pairs
    print(f"[RAGAS] CSV not found at {CSV_PATH}, using SAMPLE_QA fallback.")
    return SAMPLE_QA


def run_evaluation(
    pipeline,
    qa_pairs: Optional[List[Dict]] = None,
    top_k: int = 5,
    store: str = "faiss",
    phase: str = "current",
) -> pd.DataFrame:
    """
    Run RAGAS evaluation using VertexAI compliance classes.
    Results are saved per-phase so regressions can be tracked across phases.
    No external API keys allowed.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
    from datasets import Dataset

    llm = LangchainLLMWrapper(
        ChatVertexAI(
            model_name=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION", "us-central1"),
        )
    )

    emb = LangchainEmbeddingsWrapper(
        VertexAIEmbeddings(
            model_name="text-embedding-005",
            project=os.environ.get("GCP_PROJECT_ID"),
            location=os.environ.get("GCP_LOCATION", "us-central1"),
        )
    )

    if qa_pairs is None:
        qa_pairs = load_qa_pairs()

    questions, answers, contexts, ground_truths = [], [], [], []

    for qa in qa_pairs:
        question = qa["question"]
        answer_gen, docs, _ = pipeline.query(
            question, top_k=top_k, store=store, stream=False
        )
        answer = "".join(answer_gen)
        questions.append(question)
        answers.append(answer)
        contexts.append([d.get("content", "") for d in docs])
        ground_truths.append(qa.get("ground_truth", ""))

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for m in metrics:
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = emb

    result = evaluate(dataset, metrics=metrics)

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"phase{phase}_results.csv"
    result.to_pandas().to_csv(out_file, index=False)
    print(f"[RAGAS] Results saved to {out_file}")

    return result.to_pandas()
