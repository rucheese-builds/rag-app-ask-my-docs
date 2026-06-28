# AgentLens — Local RAG Intelligence Dashboard

AgentLens is a local, production-grade Retrieval-Augmented Generation (RAG) system designed to query a corpus of the latest academic research on **Web-of-Agents** alongside **earnings calls** from top enterprise AI companies.

---

## ⚡ Key Features

*   **Cross-Corpus Synthesis:** Ask how academic research on multi-agent collaboration maps to enterprise adoption (e.g., Salesforce's Agentforce) and get cited, synthesized answers drawing from both spaces.
*   **Source Citation Enforcement:** Answers are strictly grounded in retrieved documents, mapping claims directly back to their source document names and page numbers.
*   **Adversarial Query Classifier:** Implements a hybrid classifier gating off-domain queries or semantic traps before they reach the vector database.
*   **Sleek Modern UI:** Built with Streamlit, incorporating high-fidelity CSS styling, interactive benchmarking progress, and visualizers.

---

## 🗺️ High-Level Architecture

```
User Query
    ↓
Query Classifier (Hybrid Allowlist + Mistral 7B)
    ↓ (IN_DOMAIN)
Query Expansion Node (Mistral 7B Semantic Translation)
    ↓
Dense Search (Chroma) + Sparse Search (BM25)
    ↓
Convex Score Fusion (alpha = 0.4)
    ↓
Cross-Encoder Reranker (MiniLM-L-6-v2)
    ↓
Context Swapper (Child -> Parent Text Mapping)
    ↓
LLM Generator with Citation Enforcement (Mistral 7B)
    ↓
Cited Answer + Source Badges
```

---

## 🛠️ The Tech Stack (Runs Locally)

*   **Core Framework:** LangChain
*   **Embeddings:** `nomic-embed-text` via Ollama
*   **Vector Database:** Chroma
*   **Sparse Keyword Search:** rank-bm25 (Okapi)
*   **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
*   **Local LLM:** `mistral` via Ollama
*   **Frontend Dashboard:** Streamlit

---

## 📚 Repository Documentation Index

For in-depth explanations, trial histories, and evaluation metrics, see the dedicated documentation files:

*   📖 **[Architectural Journey & Design Decisions](docs/architecture.md):** Traces the evolution from flat chunking to parent-child indexing, hybrid query routing, convex fusion weight-tuning, and cross-encoder benchmarks.
*   📊 **[Evaluation & Benchmarking Journey](docs/evaluation.md):** Details our custom evaluation framework (metrics formulas), local LLM-as-a-judge prompts, TruLens/RAGAS package trials, adversarial datasets, and reference pruning performance.

---

## 🚀 Running Locally

### 1. Installation
Clone the repository and install the dependencies using `uv` (recommended):
```bash
git clone https://github.com/rucheese-builds/rag-app-ask-my-docs
cd rag-app-ask-my-docs
uv sync
```

### 2. Pull Local Models
Ensure you have [Ollama](https://ollama.com) installed and running, then pull the required models:
```bash
ollama pull mistral
ollama pull nomic-embed-text
```

### 3. Build the Database & Indices
Ingest, prune, and index the document corpus:
```bash
.venv/bin/python ingestion/ingest.py
```

### 4. Run the Evaluation Suite
Compute local metrics against the golden dataset:
```bash
.venv/bin/python evaluation/evaluate.py
```

### 5. Launch the Dashboard Applications
*   **Launch the Main Platform Dashboard (Port 8501):**
    ```bash
    .venv/bin/streamlit run app/app.py --server.port 8501
    ```
*   **Launch the Golden Q&A Vetting Tool (Port 8502):**
    ```bash
    .venv/bin/streamlit run tools/qa_generator/app.py --server.port 8502
    ```
