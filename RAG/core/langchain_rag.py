from __future__ import annotations
import os
from typing import List, Optional, Generator

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from observability.langfuse_client import get_langfuse_client

def get_lc_embeddings() -> VertexAIEmbeddings:
    """Initializes standard Vertex AI embeddings provider."""
    return VertexAIEmbeddings(
        model_name="text-embedding-005",
        project=os.environ.get("GCP_PROJECT_ID"),
        location=os.environ.get("GCP_LOCATION", "us-central1"),
    )

def build_langchain_retriever(faiss_store, top_k: int = 5):
    """Reuses the existing native FAISS index without re-embedding data collection."""
    from langchain_community.vectorstores import FAISS as LCFAISS
    
    embeddings = get_lc_embeddings()
    index = faiss_store.index
    
    docstore = {
        str(i): Document(
            page_content=d.get("content", ""), 
            metadata=d
        ) for i, d in enumerate(faiss_store.documents)
    }
    index_to_docstore_id = {i: str(i) for i in range(len(faiss_store.documents))}
    
    lc_vectorstore = LCFAISS(
        embedding_function=embeddings,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    return lc_vectorstore.as_retriever(search_kwargs={"k": top_k})

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert financial analyst assistant specializing in IFC annual reports.

Answer questions based ONLY on the provided context. If the information is not in the context, clearly state that you don't know.

Rules:
- Cite precise figures and always mention the fiscal year (FY24, FY23, etc.)
- For tables, present data in a structured way
- Indicate the source page number when available
- Be concise and to the point
- Return your answer as JSON with keys: answer (string), citations (list of page numbers)

Context:
{context}"""),
    ("human", "{question}"),
])

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        ctype = doc.metadata.get("content_type", "text")
        parts.append(f"[Source {i} | Page: {page} | Type: {ctype}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def build_rag_chain(retriever, stream: bool = True):
    """Builds compliant LCEL sequence supporting native JSON output constraints."""
    llm = ChatVertexAI(
        model_name=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
        project=os.environ.get("GCP_PROJECT_ID"),
        location=os.environ.get("GCP_LOCATION", "us-central1"),
        temperature=0.1,
        max_output_tokens=1024,
        streaming=stream,
        response_mime_type="application/json",
    )

    chain = (
        {
            "context": retriever | RunnableLambda(_format_docs),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

def langchain_query(
    question: str,
    faiss_store,
    top_k: int = 5,
    stream: bool = True,
) -> tuple:
    """Executes RAG pipeline via LangChain LCEL wrapper and instruments telemetry spans."""
    langfuse = get_langfuse_client()

    with langfuse.start_as_current_observation(
        as_type="span",
        name="langchain-rag-query",
        input={"question": question},
    ) as span:
        retriever = build_langchain_retriever(faiss_store, top_k=top_k)
        chain = build_rag_chain(retriever, stream=stream)

        docs = retriever.invoke(question)
        doc_dicts = [
            {
                "content": d.page_content,
                "page": d.metadata.get("page", 0),
                "content_type": d.metadata.get("content_type", "text"),
                "score": 1.0,
            }
            for d in docs
        ]

        if stream:
            def _stream_gen():
                for chunk in chain.stream(question):
                    yield chunk
            span.update(output={"num_docs": len(docs)})
            return _stream_gen(), doc_dicts
        else:
            answer = chain.invoke(question)
            span.update(output={"answer": answer[:200], "num_docs": len(docs)})
            def _gen():
                yield answer
            return _gen(), doc_dicts