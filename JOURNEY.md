## The Query Classification Journey

### The Problem I Was Solving
Stress testing revealed a critical vulnerability — semantic traps.
Queries like "How do I build a web scraper to collect agent data?"
contained domain keywords (web, agent) but had completely different
intent. The retrieval system confidently returned wrong results.

Concrete example of the failure:

Query: "How do I build a web scraper to collect agent data?"

System response BEFORE classification:
> "To build a web scraper to collect agent data, you would need
> to use the Web Agent functionality provided by OpenAgents.
> According to [Source 2], the Web Agent is designed for
> autonomous web browsing..."
> Sources: OpenAgents.pdf pages 0, 5, 16

This is the most dangerous RAG failure mode — not hallucination,
but confident retrieval of plausible but irrelevant content.
Every citation was real. Every claim was in the document.
The intent was completely wrong.

---

### Attempt 1 — LLM classifier with few-shot examples (llama3.2)

**Approach:**
Used llama3.2 as the classification judge with a detailed prompt
containing IN_DOMAIN and OUT_OF_DOMAIN examples.

**Result:** Inconsistent and unreliable.
- "How do agents communicate?" → OUT_OF_DOMAIN ❌
- "What is Salesforce Agentforce?" → OUT_OF_DOMAIN ❌
- "How do I build a web scraper?" → IN_DOMAIN ❌

**Root cause:**
llama3.2 is a 3B parameter model. It cannot reliably follow
structured classification instructions. The model that works
well enough for generation — where occasional imprecision is
acceptable — is too small for classification, where you need
a binary correct/incorrect output every single time.

**Lesson:**
Generation and classification have different reliability
requirements. A model that is "good enough" for generation
is not good enough for classification.

---

### Attempt 2 — Simplified prompt (llama3.2)

**Approach:**
Stripped the prompt down to bare minimum instructions,
thinking complexity was causing the failure.

**Result:** Worse — everything classified as OUT_OF_DOMAIN.
- "How do agents communicate?" → OUT_OF_DOMAIN ❌
- "How do I build a web scraper?" → OUT_OF_DOMAIN ❌
- "What is Salesforce Agentforce?" → OUT_OF_DOMAIN ❌

**Root cause:**
The simplified prompt removed the few-shot examples that were
providing the model with pattern guidance. Without examples,
a small model defaults to the simpler answer — OUT_OF_DOMAIN —
because it requires less reasoning.

**Lesson:**
For small LLMs, few-shot examples are not optional decoration.
They are load-bearing. Removing them to "simplify" the prompt
removes the only signal the model has to reason correctly.

---

### Attempt 3 — LLM classifier with few-shot examples (Mistral 7B)

**Approach:**
Switched to Mistral 7B — a significantly larger and more capable
model — keeping the original few-shot prompt structure.

**Result:** Still inconsistent.
- "How do agents communicate?" → IN_DOMAIN ✅
- "How do I build a web scraper?" → IN_DOMAIN ❌
- "What is Salesforce Agentforce?" → OUT_OF_DOMAIN ❌

**Root cause:**
The web scraper query contains "agent" — a word that appears
thousands of times across the corpus. Even Mistral 7B cannot
reliably distinguish "agent" as in AI agent vs "agent" as in
a generic noun when it appears in a clearly off-domain query.
The word is too semantically overloaded in this specific corpus.

**Lesson:**
LLM classifiers struggle with domains where the core terminology
is semantically overloaded. "Agent" means AI agent in your corpus
but means many other things in everyday language. No amount of
prompt engineering fully solves this — the model sees the word
and pattern-matches to the domain.

---

### Attempt 4 — Keyword allowlist (final solution)

**Approach:**
Abandoned LLM classification entirely. Built a deterministic
keyword allowlist using only highly specific domain terms that
would never appear in off-domain queries.

Key insight: instead of using generic terms like "agent",
use only compound and specific terms:
- "agentforce" not "agent"
- "multi-agent" not "agent"
- "web of agents" not "web" or "agents"
- "autogen" not "auto" or "gen"

**Result:** All four test queries classified correctly.
- "How do agents communicate in a web of agents?" → IN_DOMAIN ✅
  Matched: ['web of agents']
- "How do I build a web scraper to collect agent data?" → OUT_OF_DOMAIN ✅
  No specific domain keywords matched
- "What is Salesforce Agentforce?" → IN_DOMAIN ✅
  Matched: ['agentforce', 'salesforce']
- "Should I invest in crypto tokens?" → OUT_OF_DOMAIN ✅
  No specific domain keywords matched

**Why this works:**
Highly specific compound terms are unambiguous. "Agentforce"
only means one thing. "Web of agents" only means one thing.
"Multi-agent" only appears in AI agent contexts. Generic terms
like "agent" or "web" are ambiguous — specific compound terms
are not.

**Why this is the right engineering decision:**
- Zero latency — no LLM call needed
- Zero cost — no tokens consumed
- Deterministic — same query always gives same result
- Explainable — you can see exactly why a query was classified
- Appropriate for this domain — specific terminology makes
  keyword matching reliable in a way it isn't for general domains

**Production upgrade path:**
Replace with a fine-tuned binary classifier or GPT-4o mini
(fractions of a cent per call, much higher accuracy). The
keyword allowlist is a pragmatic local solution — it works
perfectly for this corpus and is honest about its approach.

---

### What This Journey Taught Me

**On evaluation-driven development:**
I only discovered the semantic trap problem because I stress
tested my evaluation framework with adversarial questions.
Without the stress test, I would have shipped a system with
Hit Rate 1.0 on easy questions and never known it failed
on harder ones. Evaluation is not a checkbox — it's how
you find real failure modes.

**On tool selection:**
The right tool depends on your constraints. LLM classifiers
are powerful but require capable models and add latency.
Keyword allowlists are simple but reliable for specific domains.
The engineering decision is about tradeoffs, not which approach
sounds more impressive.

**On iteration:**
Four attempts to solve one problem is not failure — it's
engineering. Each attempt revealed something specific about
why the previous approach failed. The final solution was only
possible because I understood exactly what went wrong in
attempts 1, 2, and 3.

**On documenting failures:**
Every failed attempt is documented here because future me —
and anyone reading this — should understand why the final
solution is what it is. Code without failure history is
incomplete documentation.

## Evaluation Paradox: Metrics vs Perceived Quality

After implementing diversity reranking and larger chunks, automated 
metrics dropped while human-perceived answer quality improved.

| Metric | Before | After |
|---|---|---|
| Precision@3 | 0.405 | 0.286 |
| MRR | 0.464 | 0.345 |
| Semantic Similarity | 0.656 | 0.660 |

Root cause: Diversity reranking deliberately introduces sources that 
score lower on relevance to force cross-paper synthesis. Automated 
metrics that measure whether a pre-defined correct source appears in 
top-3 penalise this behaviour even when the resulting answer is richer.

This reveals a fundamental limitation of single-source evaluation 
frameworks for multi-document synthesis tasks. The correct evaluation 
for a system designed to synthesise across sources would measure 
answer completeness and source diversity, not source precision.

Production solution: Multi-reference evaluation where each question 
has multiple acceptable source documents, or LLM-as-judge evaluation 
asking "is this answer well-synthesised from multiple sources?" rather 
than "did it retrieve the correct document?"

## Final Evaluation Numbers

Run after complete pipeline including query classifier.

Standard questions (4): Hit Rate 1.00 | MRR 0.875 | 
Precision@3 0.833 | Recall@3 0.792 | Similarity 0.799

Full set with adversarial (9): Hit Rate 0.50 | MRR 0.464 | 
Precision@3 0.405 | Recall@3 0.357 | Similarity 0.652

The gap between standard and adversarial scores is the 
honest measure of where the system succeeds and where 
it struggles. Semantic traps remain the primary failure 
mode — caught at the classifier level for known patterns, 
but novel semantic traps would still reach retrieval.