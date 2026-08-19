import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PATH = BASE_DIR / "chroma_db"
VECTOR_DB_DIR = CHROMA_PATH
EVAL_DIR = BASE_DIR / "eval"

# Chunking Configuration (Tuned via Ablation Experiment)
CHUNK_SIZE = 400       # token count estimate (x4 for character count in splitter)
CHUNK_OVERLAP = 50     # token count estimate (x4 for character count in splitter)

# Retrieval Configuration (Tuned for Clinical Decision Support)
TOP_K = 4              # Balanced top_k ensuring high coverage without context dilution

# Embedding Model Configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BENCHMARK_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2"
]
