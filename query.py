import sys
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma

import config
from ingest import get_embedding_function


def load_index(
    persist_directory: Optional[str] = None,
    embedding_function = None,
    collection_name: Optional[str] = "diagnosis_red_flags"
) -> Chroma:
    """
    Loads an existing Chroma vector database from disk.
    """
    if persist_directory is None:
        persist_directory = str(config.CHROMA_PATH)
    if embedding_function is None:
        embedding_function = get_embedding_function()

    kwargs = {
        "persist_directory": persist_directory,
        "embedding_function": embedding_function
    }
    if collection_name:
        kwargs["collection_name"] = collection_name

    return Chroma(**kwargs)


def retrieve(
    vectordb: Chroma,
    query: str,
    k: int = 4,
    score_threshold: Optional[float] = None
) -> List[Tuple[Document, float]]:
    """
    Retrieves top-k documents with cosine relevance scores.
    Optionally filters by score_threshold.
    """
    results = vectordb.similarity_search_with_relevance_scores(query, k=k)
    if score_threshold is not None:
        results = [(doc, score) for doc, score in results if score >= score_threshold]
    return results


def format_retrieval_for_explainability(
    results: List[Tuple[Document, float]],
    query: Optional[str] = None
) -> str:
    """
    Generates a structured clinical explainability report for retrieved evidence chunks,
    displaying rank, relevance score, source document, page citation, chunk ID, and excerpt.
    """
    lines = []
    if query:
        lines.append(f"Query: {query}")
        lines.append("=" * 80)
    lines.append(f"{'Rank':<5} {'Score':<8} {'Citation / Source':<35} {'Chunk ID'}")
    lines.append("-" * 80)

    for i, (doc, score) in enumerate(results, 1):
        doc_name = doc.metadata.get("document_name", "Unknown")
        page_num = doc.metadata.get("page_number", "?")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        citation = f"{doc_name} (p. {page_num})"
        
        lines.append(f"[{i}]   {score:<8.3f} {citation:<35} {chunk_id}")
        # Clean excerpt
        clean_text = " ".join(doc.page_content.split())
        lines.append(f"      Evidence: \"{clean_text[:160]}...\"\n")

    return "\n".join(lines)


if __name__ == "__main__":
    query_text = sys.argv[1] if len(sys.argv) > 1 else "What are the red flags of cauda equina syndrome?"
    print(f"\nLoading index from {config.CHROMA_PATH}...")
    db = load_index()
    
    print(f"\nExecuting clinical query: '{query_text}' (top_k={config.TOP_K})\n")
    retrieved = retrieve(db, query_text, k=config.TOP_K)
    
    print(format_retrieval_for_explainability(retrieved, query=query_text))
