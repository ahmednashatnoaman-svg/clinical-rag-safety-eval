# Clinical RAG Safety & Internal Evaluation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/Chroma-Vector_DB-orange)](https://www.trychroma.com/)

A robust implementation of safety guardrails and internal evaluation metrics for a Clinical Retrieval-Augmented Generation (RAG) system. This repository focuses on the operational reliability required for clinical applications, specifically addressing confidence calibration and ungrounded claim detection (hallucination prevention).

## Core Architecture

This project implements a dual-layer safety mechanism to ensure the generated medical advice is strictly backed by the retrieved evidence.

```mermaid
graph TD
    A[User Query] --> B[Retrieve Documents (Chroma)]
    B --> C{Confidence Check}
    C -- "Score < Calibrated Threshold" --> D[Refusal Response]
    C -- "Score >= Threshold" --> E[Generate Answer]
    E --> F[Extract Generated Claims]
    F --> G{Unsupported Claim Detector}
    G -- "Lexical Overlap < 35%" --> H[Flag/Refusal]
    G -- "Lexical Overlap >= 35%" --> I[Final Answer Delivery]

    classDef critical fill:#f9f,stroke:#333,stroke-width:2px;
    class C,G critical;
```

### 1. Calibrated Confidence Threshold
Instead of arbitrarily selecting a similarity threshold to determine if a question is answerable, we calibrate the threshold empirically based on the distribution of retrieval scores.
- **Answerable Queries**: Questions that map directly to the index.
- **Unanswerable/Out-of-Domain Queries**: Queries with no relevant grounding in the source material.
- **Methodology**: We define the threshold either at the midpoint of the empirical score gap (if one exists) or at the point of highest classification accuracy. This strictly bounds when the system should default to a safe refusal.

### 2. Lexical Overlap Safety Net
Even with optimal prompting, language models may drift and inject hallucinated specifics (e.g., specific dosages not mentioned in the text). 
- **Mechanism**: The generated recommendation is parsed into distinct claims.
- **Verification**: Each claim's substantive vocabulary (excluding stopwords) is intersected with the retrieved evidence. If the overlap is below `35%`, the claim is flagged as unsupported.
- **Result**: Provides a second, independent deterministic safety layer that guards against LLM drift.

## Evaluation Metrics

This repository implements automated benchmarks across a set of clinical retrieval cases and deliberate safety cases:

- **Average Precision@K (Precision@3)**: Verifies that the correct expected document page is successfully retrieved in the top K results.
- **Safety Pass Rate**: The percentage of safety/out-of-domain test cases where the system correctly triggered a refusal.

## Project Structure

```
├── Task4_Safety_Evaluation.ipynb # Core notebook executing safety calibration and metrics
├── data/                         # Expected source guidelines (e.g., WHO Hypertension)
├── eval/                         # Day4_Starter_Benchmark.csv containing test cases
├── config.py                     # Centralized configuration (Paths, Thresholds, Models)
├── ingest.py                     # Deterministic PDF chunking and metadata stamping
├── query.py                      # RAG retrieval module
├── JUSTIFICATION.md              # Architectural decisions for safety evaluation
└── README.md                     # Project documentation
```

## Running the Evaluation

Ensure your environment has the required dependencies installed (LangChain, ChromaDB, FastEmbed, PyPDF).

1. Execute the `Task4_Safety_Evaluation.ipynb` notebook.
2. The notebook will automatically:
   - Rebuild the local index from the provided `data/` directory.
   - Run the calibration sets to determine the `CONFIDENCE_THRESHOLD`.
   - Test the unsupported-claim detector on a controlled drift case.
   - Run the final benchmark, outputting the `Average Precision@3` and `Safety Pass Rate`.

## Disclaimer
This project is an architectural demonstration of safety mechanisms for LLM-based clinical decision support. The system and its outputs are not intended to serve as real-world medical advice.
