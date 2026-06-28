# Architectural Journey of AgentLens

This document details the engineering flow, key architectural decisions, trialled configurations, and the evolution of the AgentLens local RAG system.

---

## 🗺️ Engine Architecture Flow

```
User Query
    ↓
Query Classifier (Hybrid Keyword + LLM check) — Is this in-domain?
    ↓ yes                                              ↓ no
Query Expansion Node (Mistral 7B Translation)      "Outside my corpus"
    ↓
Search Namespace Filter (All / Research / Earnings)
    ↓
Dense Search (Vector, k=20) + Sparse Search (BM25, k=20)
    ↓
Convex Score combination (Alpha-tuned Score Fusion)
    ↓
Cross-Encoder Reranking (MiniLM-L-6-v2) ➔ Top 3 child chunks
    ↓
Context Swapper (Map matched child ➔ parent chunk text)
    ↓
LLM Generation with Citation Enforcement (Mistral 7B)
    ↓
Cited Answer + Source List
```

---

## 🛠️ Key Architectural Decisions

### 1. Hierarchical Parent-Child Indexing
* **Concept:** Splitting documents into dual-tier chunks rather than flat, uniform pieces.
* **Recursive Flat Chunking Details:** In our initial exploration, the baseline "flat chunking" strategy was implemented using `RecursiveCharacterTextSplitter` (splitting by paragraphs, newlines, and sentences to avoid cutting words/phrases in half), rather than a rigid character-based splitter.
* **The Failure Mode of Flat Chunking (The Size-vs-Density Trade-off):** We did not run a dedicated comparison between recursive flat chunking and character flat chunking (rigid split) because the primary bottleneck of the flat chunking approach was not the splitting algorithm itself, but rather the **fundamental size-vs-density trade-off**:
  * **Small Flat Chunks (e.g., 400 characters):** Excelled at vector search because the semantic matches were dense and specific (improving **Hit Rate** and **Context Precision**). However, once sent to the LLM, they lacked surrounding context (e.g., splitting a table or cutting off reasoning steps), leading to poor **Faithfulness** and **Answer Relevance**.
  * **Large Flat Chunks (e.g., 1500 characters):** Provided complete context to the LLM for high-quality generation, but suffered from **embedding dilution** (where specific technical terms or numeric data got averaged out in the 768-dimension vector space). This caused a drop in retrieval metrics (**Hit Rate** and **Recall**).
* **The Fix / Resolution:** Switched to **Hierarchical Parent-Child Indexing** (implemented in [ingest.py](file:///Users/ruchiagarwal/.gemini/antigravity/scratch/rag-app-ask-my-docs/ingestion/ingest.py#L61-L112)), where both tiers utilize `RecursiveCharacterTextSplitter`. We vectorize dense, 400-character **child chunks** for vector search, but retrieve and feed their corresponding 1500-character **parent chunks** to the LLM at generation time. This yields both high retrieval precision and complete synthesis context.

---

### 2. Hybrid Query Classification
* **Concept:** A gateway classifier to detect and block out-of-domain queries before they trigger database retrieval.
* **The Failure Mode (Semantic Traps):** Adversarial queries containing domain keywords with off-domain intent (e.g., *"How do I build a web scraper to collect agent data?"*) triggered false positive retrievals from the vector database, leading the LLM to generate cited, plausible-looking, but incorrect answers.
  * **Scraper Query Failure Example (Before Fix):**
    > **Query:** *"How do I build a web scraper to collect agent data?"*
    > **Answer:** *"To build a web scraper to collect agent data, you would need to use the Web Agent functionality provided by OpenAgents. According to [Source 2], the Web Agent is designed for autonomous web browsing..."*
    > **Sources cited:** `OpenAgents.pdf` pages 0, 5, 16.
  * **Why this is the most dangerous failure mode:** There was zero hallucination and every citation was real. However, the intent was completely wrong. Because the keywords `web` and `agent` appeared heavily in the corpus, the vector store pulled a paper about AI web-browsing agents. This proved that **citation enforcement cannot save you from wrong retrieval**.
* **Trialled & Replaced LLM Classifier Solutions:**
  * **Attempt 1 (few-shot Llama 3.2 3B):** Too small to follow structured binary classification.
  * **Attempt 2 (simplified Llama 3.2 prompt):** Removing examples broke small LLM reasoning, causing it to classify everything as out-of-domain.
  * **Attempt 3 (few-shot Mistral 7B):** Better, but still failed on terms like "agent" that are heavily overloaded across the corpus.
* **The Final Fix / Hybrid Classifier:** We implemented a hybrid classification strategy (implemented in [classifier.py](file:///Users/ruchiagarwal/.gemini/antigravity/scratch/rag-app-ask-my-docs/generation/classifier.py)) that marries deterministic keywords with an LLM fallback:
  * **Phase 1: Fast-Path Static Keyword Allowlist (Zero-Latency):** The system first checks the query against a list of 21 highly specific domain keywords (such as `camel`, `autogen`, `agentforce`, `multi-agent`). If any match is found, the query is immediately flagged as `IN_DOMAIN` with zero latency and zero token cost.
  * **Phase 2: LLM Semantic Fallback (Mistral 7B):** If there are no keyword matches, the system invokes a local `Mistral 7B` model as a fallback classifier with a corpus scope prompt. This ensures that semantic queries that don't use exact allowed keywords but are still within domain (e.g., *"How do distributed software systems collaborate?"*) are still correctly permitted. If the LLM call fails, the system defaults to `True` for safety.

---

### 3. Sparse-Dense Hybrid Retrieval: Convex Score Fusion
* **Concept:** Merging sparse (BM25) and dense (Vector) search spaces.
* **Language Gap across Corpus:** Research papers use formal academic language. Earnings calls use business and product language. The same concepts have completely different vocabularies. BM25 captures exact terminology, while Vector search captures semantic similarity.
* **Why Reciprocal Rank Fusion (RRF) was Replaced:** RRF sorts results purely based on rank order:
  $$S_{RRF} = \sum_{m} \frac{1}{k + \text{rank}_m}$$
  RRF is blind to score margins and cannot be tuned to prioritize exact matches. We replaced it with **Convex Score Fusion**:
  $$S_{hybrid} = \alpha \cdot S_{dense} + (1 - \alpha) \cdot S_{sparse}$$
  Setting $\alpha = 0.4$ prioritizes precise keyword matches (e.g., specific terms or acronyms like *DyLAN*, *L2M2*, *Agentforce*) in the BM25 space while retaining vector semantic recall.

---

### 4. Cross-Encoder Reranking
* **Concept:** Re-assessing candidate relevance by encoding the query and document together.
* **Bi-Encoders vs. Cross-Encoders:** Bi-encoders (like `nomic-embed`) encode the query and document separately into static vector spaces, making retrieval extremely fast but less accurate. Cross-encoders (like `ms-marco-MiniLM-L-6-v2`) feed the query and document together into the transformer, capturing deep query-document interactions.
* **Benchmarking Latency and Recall:** Bypassing the reranker to feed chunks directly from Convex Fusion was tested to reduce latency, but results showed:
  
  | Configuration | Hit Rate | MRR | Latency | Semantic Similarity |
  | :--- | :---: | :---: | :---: | :---: |
  | **With Reranker** | **0.7500** | **0.3542** | **19.14s** | **0.8305** |
  | **Without Reranker** | 0.6250 | 0.3438 | 26.13s | 0.8236 |
  | **Delta** | **+0.1250** | **+0.0104** | **-6.99s** | **+0.0069** |

* **Key Insight:** Bypassing the reranker dropped the Hit Rate by **12.5%** and was **6.99 seconds slower** on average. Without the reranker filtering out noise, the LLM spent more time/tokens reasoning over unhelpful material.

---

### 5. Automatic References & Appendix Pruning
* **Concept:** Truncating academic papers to remove citations and bibliography lists.
* **The Failure Mode:** The vector store indexed pages of raw bibliographic names, leading keyword and dense queries to mistakenly retrieve citations lists as relevant context.
* **The Fix:** Created a regex parser in `ingestion/parser.py` that scans research papers starting from page 3, detects headings matching `References` or `Bibliography`, and discards all subsequent pages.
* **Result:** Decreased academic text parsed by **50.1%** (473 to 236 pages) and indexed nodes by **30.5%** (9,499 to 6,603 chunks), speeding up index building and improving retrieval accuracy.

---

## 🔮 Future Production Roadmap (What to Do Differently)

If moving this local prototype to a production environment, the following architectural upgrades would be prioritized:

1.  **Frontier Model Integration:** Move from local Ollama (`Mistral`, `Llama 3.2`) to APIs like Gemini 1.5 Pro or GPT-4o for stronger multi-hop reasoning, lower latency, and production-grade evaluation stability.
2.  **Dedicated Metadata Namespace Filtering:** Move Chroma to a production vector database (like Qdrant or pgvector) that supports hardware-level metadata namespace pre-filtering. Currently, namespace filtering is handled at the query execution level, whereas pre-filtering at the database index level reduces search complexity.
3.  **Dynamic Query Routing:** Extend the query classifier into a multi-class router to dynamically choose whether a query needs vector search, BM25 keyword search, or structured database lookups (for strict quantitative data from earnings tables).
4.  **Streaming & Async Engine:** Implement token-by-token streaming on the Streamlit UI to improve user-perceived latency (TTFT - Time to First Token) while local inference runs.
5.  **Human-in-the-Loop Feedback:** Add simple UI thumbs-up/down buttons to capture user corrections and feed them back into the evaluation dataset.
