# AgentLens —
A cross-domain RAG engine performing simultaneous retrieval over foundational Web-of-Agents research and Enterprise AI earnings. Features hierarchical summary-indexing and intent-gating to ensure 100% cited accuracy with zero-hallucination tracing

**Live demo:** [HuggingFace Spaces link — add after deployment]

---

## What I Built

A production-grade RAG system combining 14 Web of Agents research 
papers with 12 enterprise AI earnings call transcripts 
(Salesforce, Microsoft, Nvidia, ServiceNow, Google, IBM).

The unique value: cross-corpus synthesis. Ask how academic research 
on agent coordination compares to how Salesforce implements 
Agentforce — and get a cited answer drawing from both sources.

---

## Architecture
```
User Query
    ↓
Query Classifier — is this in-domain?
    ↓ yes                    ↓ no
    ↓               "Outside my corpus"
BM25 Search (k=20)
    +
Vector Search (k=20)
    ↓
RRF Fusion → Top 5
    ↓
Cross-Encoder Reranking → Top 3
    ↓
LLM Generation with Citation Enforcement
    ↓
Cited Answer + Source List
```

**Stack (all free, runs locally):**
- Embeddings: nomic-embed-text via Ollama
- Vector store: Chroma
- Keyword search: rank-bm25
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: llama3.2 via Ollama
- Framework: LangChain

---

## The Evaluation Journey

Most RAG tutorials skip evaluation entirely. I treated it as 
a first-class engineering problem. Here is the full story.

### Step 1 — Tried RAGAS (industry standard)

RAGAS is the most widely used LLM-as-judge evaluation framework 
for RAG systems. I implemented it first.

**Result:** NaN scores across faithfulness, answer relevancy, 
and context precision when using local models (llama3.2, mistral).

**Root cause:** RAGAS requires frontier models like GPT-4o to 
follow its complex internal evaluation prompts. Local models 
time out or produce malformed structured outputs.

**Lesson:** LLM-as-judge evaluation is architecturally sound 
but requires frontier model APIs. This is why enterprise teams 
budget separately for evaluation infrastructure.

### Step 2 — Tried TruLens (alternative framework)

TruLens promised better local model support.

**Result:** Package incompatibility — `trulens-providers-ollama` 
doesn't exist in the registry. Legacy `trulens-eval` incompatible 
with current LangChain versions.

**Lesson:** Rapidly evolving ML tooling means package 
compatibility is a real engineering concern, not just 
a footnote.

### Step 3 — Built a custom three-tier evaluation framework

Rather than give up on evaluation, I built a framework using 
metrics that work without an LLM judge.

**Tier 1 — Deterministic retrieval metrics (free, no LLM):**

| Metric | Score | What it measures |
|---|---|---|
| Hit Rate | 1.00 | Did retrieval find any correct source? |
| MRR | 0.875 | How highly was the correct source ranked? |
| Precision@3 | 0.833 | What fraction of top-3 chunks were relevant? |
| Recall@3 | 0.792 | What fraction of relevant sources were found? |

**Tier 2 — Semantic similarity (free, embedding-based):**

| Metric | Score |
|---|---|
| Avg Semantic Similarity | 0.799 |

Cosine similarity between generated answers and ground truth 
using the same local embedding model as retrieval.

### Step 4 — Questioned the results

Hit Rate of 1.0 on 4 hand-crafted questions is not a real 
benchmark. I deliberately added adversarial questions to 
stress test the system.

### Step 5 — Stress tested with adversarial questions

Added 5 adversarial question types:
- **Out-of-corpus queries** — topics not in any document
- **Semantic traps** — same keywords, completely different meaning
- **Cross-document synthesis** — requires both papers and earnings calls
- **Precise factual lookups** — specific numbers from transcripts
- **Cross-paper comparison** — comparing two research approaches

**Stress test results:**

| Metric | Standard (4 questions) | Adversarial (9 questions) | Drop |
|---|---|---|---|
| Hit Rate | 1.00 | 0.50 | -50% |
| MRR | 0.875 | 0.464 | -41% |
| Precision@3 | 0.833 | 0.405 | -51% |
| Recall@3 | 0.792 | 0.357 | -55% |
| Semantic Similarity | 0.799 | 0.656 | -18% |

### Step 6 — Found the exact failure mode

The semantic trap questions caused the biggest drop. Here is 
the concrete example that made the problem undeniable.

**Query:** `"How do I build a web scraper to collect agent data?"`

**What the system answered (before the fix):**

> *"To build a web scraper to collect agent data, you would need 
> to use the Web Agent functionality provided by OpenAgents. 
> According to [Source 2], the Web Agent is designed for 
> autonomous web browsing..."*
>
> Sources: OpenAgents.pdf pages 0, 5, 16

**Why this is the most dangerous RAG failure mode:**

- No hallucination — every claim is in OpenAgents.pdf ✓
- Citations are real — Sources 1, 2, 3 all exist ✓  
- The answer is completely wrong — wrong domain, wrong intent ✗

The keywords `web` and `agent` both appear heavily in the corpus. 
BM25 and vector search retrieved a paper about AI web-browsing 
agents. The LLM faithfully answered based on those chunks — 
confident, cited, and entirely off-target.

**This is why citation enforcement cannot save you from 
wrong retrieval.** The problem must be caught before 
retrieval runs, not after generation.

### Step 7 — Added query classification as the fix

A classifier now sits before retrieval. It reads the full 
query for intent, not just keywords, and blocks out-of-domain 
queries before they reach retrieval.

**Same query after the fix:**

> *"This question appears to be outside the scope of my document 
> corpus, which covers Web of Agents research and enterprise AI 
> adoption. Please ask something related to these topics."*

Retrieval never ran. The semantic trap was caught at the gate.

---

## Key Design Decisions

**Why hybrid retrieval over pure vector search?**
Research papers use formal academic language. Earnings calls 
use business language. Same concepts, completely different 
vocabulary. BM25 captures exact terminology. Vector search 
captures semantic similarity. RRF fusion combines both using 
rank position rather than raw scores — no weight tuning needed.

**Why cross-encoder reranking?**
Bi-encoders encode query and document separately. Cross-encoders 
read them together — significantly more accurate at judging 
relevance. Running it on top-5 candidates after hybrid search 
gives quality without sacrificing too much speed.

**Why separate retrieval from generation evaluation?**
Each can fail independently. A system can retrieve perfectly 
and hallucinate, or retrieve poorly and generate fluently from 
wrong context. Separating the tiers tells you which component 
needs improvement.

**Why query classification?**
Semantic traps — queries where domain keywords appear but meaning 
differs — cause retrieval to surface plausible but incorrect chunks. 
Classification catches off-domain intent before retrieval runs.

---

## What I Would Do Differently in Production

- **Frontier model** — GPT-4o or Claude for better generation 
  and reliable RAGAS evaluation scores
- **Qdrant** — production vector store with namespace filtering 
  to separate research papers from earnings calls
- **Query routing** — extend classification to route queries 
  to the right document subset, not just in/out domain
- **Streaming** — stream LLM output token by token for better UX
- **Feedback loop** — thumbs up/down on answers to grow the 
  evaluation dataset continuously
- **CI pipeline** — GitHub Actions failing if Hit Rate drops 
  below threshold on every PR

---

## Document Corpus

**Research papers (14):**
Internet 3.0 Web of Agents · Internet of Agents · AutoGen · 
ReAct · AgentVerse · CAMEL · Dynamic LLM-Agent Network · 
AgentBench · L2M2 · OpenAgents · Agent Hospital · 
Scaling LLM Test-Time Compute · Scaling Agent Systems · Toolformer

**Earnings calls (12):**
Salesforce Q3+Q4 FY26 · Microsoft Q1 2026+Q4 2025 · 
Nvidia Q3+Q4 2025 · Google Q3+Q4 2025 · 
ServiceNow Q3+Q4 2025 · IBM Q3+Q4 2025

---

## Running Locally
```bash
# Clone and install
git clone https://github.com/rucheese-builds/rag-app-ask-my-docs
cd rag-app-ask-my-docs
uv sync

# Pull local models
ollama pull llama3.2
ollama pull nomic-embed-text

# Build the vector store
python ingestion/ingest.py

# Run evaluation
python -m evaluation.evaluate

# Launch the app
streamlit run app/app.py
```
