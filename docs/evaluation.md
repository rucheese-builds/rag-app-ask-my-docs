# Evaluation & Benchmarking Journey

This document tracks our RAG evaluation methodology, the metrics formulas, the custom LLM-as-a-judge architecture, trialled framework packages, and benchmark improvements.

---

## 📐 Evaluation Metrics Definitions & Calculations

Our system uses a custom evaluation suite running on `golden_dataset.csv`. The metrics are calculated as follows:

### 1. Retrieval Metrics

*   **Hit Rate (Recall)**: Measures whether the correct source document is in the top-$K$ chunks.
    $$\text{Hit Rate} = \begin{cases} 1.0 & \text{if } \text{relevant\_source} \in \text{retrieved\_sources} \\ 0.0 & \text{otherwise} \end{cases}$$
*   **Mean Reciprocal Rank (MRR)**: Measures where the first correct document is ranked:
    $$\text{MRR} = \frac{1}{\text{rank of first relevant chunk}}$$

---

### 2. Overlap & Ground Truth Similarity

*   **Fact Recall (Keyword Coverage)**: Checks if mandatory keywords from the reference answer are present in the response.
    $$\text{Fact Recall} = \frac{\text{Matched Keywords}}{\text{Total Mandatory Keywords}}$$
*   **Semantic Similarity**: Computes the Cosine Similarity of vectors embedding both answers (using `nomic-embed-text`):
    $$\text{Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$

---

### 3. LLM-as-a-Judge Metrics
Calculated via local `Mistral 7B` at `temperature=0.0`:

*   **Faithfulness**: Measures if the answer is grounded strictly in the context, preventing hallucinations. The model extracts individual claims from the answer and checks if they exist in the context:
    $$\text{Faithfulness} = \frac{\text{Claims supported by context (YES)}}{\text{Total claims identified}}$$
*   **Answer Relevance**: Evaluates how directly the answer addresses the question (penalizing fluff or off-topic information) on a scale of `0.0` to `1.0`.
*   **Context Precision**: Evaluates the relevance of the retrieved context using Precision@k:
    $$\text{Context Precision} = \frac{\sum_{k} (\text{Precision@k} \times \text{Relevance}_k)}{\text{Total relevant chunks}}$$
*   **Context Recall**: Measures if all facts in the reference answer are present in the retrieved context:
    $$\text{Context Recall} = \frac{\text{Facts found in context}}{\text{Total facts in reference answer}}$$

---

## 🔬 Trialled Evaluation Frameworks (Not Shipped)

### 1. RAGAS (Retrieval Augmented Generation Assessment)
*   **Approach**: Use the industry-standard RAGAS library to evaluate generation quality locally.
*   **Why it was replaced**: RAGAS prompts are highly complex and optimized for GPT-4. When run against local models (like `Llama 3.2` or `Mistral`), the models outputted malformed JSON or timed out, yielding `NaN` scores.
*   **Key Lesson**: LLM-as-a-judge evaluation is architecturally sound but requires frontier model APIs. This is why enterprise teams budget separately for evaluation infrastructure.

### 2. TruLens
*   **Approach**: Use TruLens for evaluation tracking.
*   **Why it was replaced**: Version incompatibilities between the latest LangChain libraries and TruLens caused dependency conflicts. Ollama support in TruLens was undocumented or stale at the time of trial.
*   **Key Lesson**: Rapidly evolving ML tooling means package compatibility is a real engineering concern, and just documentation can't be trusted.

---

## ⚡ Adversarial Stress Testing

Evaluating with simple, hand-crafted queries creates a false sense of security. To stress-test the system, we introduced 5 adversarial question types:

1.  **Out-of-corpus queries**: Asking off-topic questions (e.g., about cryptocurrency) to check if the classifier blocks them.
2.  **Semantic traps**: Queries containing key corpus vocabulary with unrelated intent (e.g., *"How do I build a web scraper to collect agent data?"*).
3.  **Cross-document synthesis**: Queries requiring information from both academic papers and earnings transcripts.
4.  **Precise factual lookups**: Specific percentage rates or deal metrics from earnings reports.
5.  **Cross-paper comparisons**: Directly comparing architectural features of two distinct agent frameworks.

### Stress Test Benchmark Results (Standard vs. Adversarial)

Before the query classification fix, the introduction of adversarial questions led to a massive drop in retrieval and generation scores, revealing the system's core vulnerabilities:

| Metric | Standard (4 questions) | Adversarial (9 questions) | Drop |
| :--- | :---: | :---: | :---: |
| **Hit Rate** | 1.00 | 0.50 | **-50%** |
| **MRR** | 0.875 | 0.464 | **-41%** |
| **Precision @ 3** | 0.833 | 0.405 | **-51%** |
| **Recall @ 3** | 0.792 | 0.357 | **-55%** |
| **Semantic Similarity** | 0.799 | 0.652 | **-18%** |

---

## 📈 Baseline Metrics Comparison: Reference Pruning

Implementing the automatic references pruner (neutralizing bibliography pages) yielded the following metric improvements on the golden dataset:

| Metric | Before References Pruning | After References Pruning | Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Avg Hit Rate (Recall)** | 0.6429 | **0.7143** | **+7.14%** | 🟢 Improved |
| **Avg Recall @ 3** | 0.5238 | **0.5833** | **+5.95%** | 🟢 Improved |
| **Avg Answer Relevance** | 0.8643 | **0.9250** | **+6.07%** | 🟢 Improved |
| **Avg Context Precision** | 0.1488 | **0.2321** | **+8.33%** | 🟢 Improved |
| **Avg Semantic Similarity** | 0.7975 | **0.8222** | **+2.47%** | 🟢 Improved |
| **Avg Faithfulness** | 0.9821 | 0.8929 | -8.92% | 🟡 Minor Shift |
| **Avg Context Recall** | 0.6307 | 0.5593 | -7.14% | 🟡 Minor Shift |

---

## 🧩 The Evaluation Paradox: Metrics vs. Perceived Quality

During architectural optimization, we encountered a RAG evaluation paradox. After implementing **diversity reranking** (forcing retrieval of diverse documents to support cross-document synthesis), our automated metrics dropped, while human-perceived answer quality improved:

| Metric | Before Diversity Reranking | After Diversity Reranking | Delta |
| :--- | :---: | :---: | :---: |
| **Precision @ 3** | 0.405 | 0.286 | **-11.9%** |
| **MRR** | 0.464 | 0.345 | **-11.9%** |
| **Semantic Similarity** | 0.656 | 0.660 | **+0.4%** |

*   **Root Cause**: Diversity reranking intentionally selects chunks from secondary papers that might have lower individual query-similarity scores in order to synthesize a broader answer. Single-source metrics (which penalize the system if top chunks are not from a single "correct" document) score this as a failure, even though the resulting synthesized answer is richer and more complete.
*   **Key Lesson**: Traditional single-source RAG metrics penalize multi-source synthesis. Evaluation of synthesis tasks must rely more heavily on LLM-as-a-judge metrics (Faithfulness and Context Recall) rather than standard document matching.

---

## 🧱 Why Separate Retrieval from Generation Evaluation?

In our custom framework, we explicitly separated the retrieval stage evaluation from the generation stage evaluation. 

*   **Reasoning**: Retrieval and generation can fail independently. 
    *   A system can retrieve the correct context perfectly but fail during generation (e.g., hallucinating details or omitting facts).
    *   Alternatively, a system can retrieve completely wrong context but generate a fluent, grammatically correct answer from that incorrect context.
*   **Engineering Value**: By isolating the stages, we can pinpoint exactly which part of the pipeline needs tuning. If Hit Rate is high but Faithfulness is low, the issue lies in LLM prompt instructions. If Hit Rate is low, we tune weights ($\alpha$) or adjust parsing boundaries rather than tweaking generation prompts.
