# Architectural Justifications

## 1. Calibrated Confidence Threshold

### Why Calibrate Empirically?
A common anti-pattern in RAG systems is hardcoding an arbitrary similarity score threshold (e.g., `0.75`) to determine if a question is answerable. This approach fails because embedding models do not output absolute, universally bounded probabilities—similarity scores are relative to the specific embedding space and domain.

### Our Approach
We implemented an empirical calibration method using a controlled set of queries:
- **Positive Control**: Queries explicitly covered by the text.
- **Negative Control**: Queries tangentially related but unanswerable by the text.

By plotting the retrieval scores of these two sets, we can:
1. Validate that our embedding model separates the classes (validating chunking/embedding upstream).
2. Calculate a precise separation boundary (either the midpoint of the gap or the threshold maximizing classification accuracy).
This transforms the "refusal mechanism" from a heuristic into a statistically backed decision boundary, minimizing both hallucinations (false positives) and unnecessary refusals (false negatives).

## 2. Lexical Overlap Safety Net

### Why an Independent Safety Net?
Prompt engineering—even rigorous grounding instructions (e.g., "Answer ONLY using the provided text")—is subject to instruction drift, particularly with complex or lengthy context windows. The model may seamlessly blend retrieved facts with pre-trained knowledge, fabricating specifics like unmentioned dosages or contraindications that appear plausible to a human reviewer.

### Our Approach
We developed a secondary, deterministic layer to detect unsupported claims:
- **Mechanism**: The LLM's generated response is split into sentence-level claims. Each claim undergoes a lexical comparison (using substantive vocabulary) against the retrieved text.
- **Why 35% Overlap?**: A strict 100% overlap requires exact extractive quotes, reducing the usefulness of an LLM. A 35% threshold strikes a balance—it allows the LLM to synthesize and rephrase information while aggressively catching hallucinations that introduce novel, out-of-context terminology. 
- **Impact**: This decoupled evaluation mechanism guarantees that even if the prompt fails, the system safely triggers a refusal rather than serving hallucinated clinical advice.
