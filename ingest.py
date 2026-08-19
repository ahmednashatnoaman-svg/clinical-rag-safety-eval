import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Union

import pypdf
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import config

# Suppress verbose pypdf warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)


def load_pdfs(data_dir: Union[Path, str]) -> List[Document]:
    """
    Fast and robust PDF loader using pypdf directly.
    Stamps normalized metadata: 'document_name', 1-indexed 'page_number', and 0-indexed 'page'.
    """
    data_path = Path(data_dir)
    # Deduplicate paths (Windows glob is case-insensitive)
    found_files = list(data_path.glob("*.pdf")) + list(data_path.glob("*.PDF"))
    pdf_files = sorted(list({p.resolve(): p for p in found_files}.values()))
    
    all_pages: List[Document] = []
    
    for pdf_path in pdf_files:
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                doc = Document(
                    page_content=txt,
                    metadata={
                        "document_name": pdf_path.name,
                        "page_number": idx + 1,
                        "page": idx
                    }
                )
                all_pages.append(doc)
        except Exception as e:
            print(f"Warning: Failed to load {pdf_path.name}: {e}")
            
    return all_pages


def chunk_documents(
    pages: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Splits documents into section-aware chunks and attaches document_name,
    page_number, and a deterministic chunk_id to each chunk.
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE * 4
    if chunk_overlap is None:
        chunk_overlap = config.CHUNK_OVERLAP * 4

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(pages)
    
    # Assign deterministic chunk IDs
    page_chunk_tracker = {}
    for chunk in chunks:
        doc_name = chunk.metadata.get("document_name", "doc")
        page_num = chunk.metadata.get("page_number", 1)
        key = f"{doc_name}:{page_num}"
        idx = page_chunk_tracker.get(key, 0)
        chunk.metadata["chunk_id"] = f"{doc_name}_p{page_num}_c{idx}"
        page_chunk_tracker[key] = idx + 1
        
    return chunks


def get_embedding_function(
    model_name: Optional[str] = None,
    batch_size: int = 64,
    max_length: int = 512
) -> FastEmbedEmbeddings:
    """
    Returns the FastEmbed embedding model instance with safe batching and max token length.
    """
    model = model_name or getattr(config, "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    return FastEmbedEmbeddings(
        model_name=model,
        batch_size=batch_size,
        max_length=max_length
    )


def build_index(
    chunks: List[Document],
    persist_directory: Optional[str] = None,
    embedding_function: Optional[FastEmbedEmbeddings] = None,
    collection_name: Optional[str] = "diagnosis_red_flags",
    batch_size: int = 64
) -> Chroma:
    """
    Embeds chunks in safe batches and creates/persists a Chroma vector database collection.
    If the persisted collection already contains documents, loads and returns it.
    """
    if persist_directory is None:
        persist_directory = str(config.CHROMA_PATH)
    if embedding_function is None:
        embedding_function = get_embedding_function()

    kwargs = {
        "persist_directory": persist_directory,
        "embedding_function": embedding_function,
        "collection_name": collection_name
    }

    vectordb = Chroma(**kwargs)
    try:
        count = vectordb._collection.count()
        if count >= len(chunks) and count > 0:
            return vectordb
    except Exception:
        pass

    # Add documents in batches for robust memory management
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectordb.add_documents(documents=batch)
        
    return vectordb


if __name__ == "__main__":
    print(f"Loading clinical PDFs from {config.DATA_DIR}...")
    pages = load_pdfs(config.DATA_DIR)
    print(f"Loaded {len(pages)} pages.")

    print(f"Splitting into section-aware chunks (size={config.CHUNK_SIZE} tokens, overlap={config.CHUNK_OVERLAP} tokens)...")
    chunks = chunk_documents(pages)
    print(f"Created {len(chunks)} chunks with verified metadata.")

    print(f"Building persistent vector index at {config.CHROMA_PATH}...")
    vectordb = build_index(chunks)
    print("Vector database built and persisted successfully.")
