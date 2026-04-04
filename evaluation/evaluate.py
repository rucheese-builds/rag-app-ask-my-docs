import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from retrieval.retriever import load_vector_store, build_bm25_index, hybrid_search
from reranking.reranker import load_reranker, rerank
from generation.generator import load_llm, generate_answer, format_context
import csv

TEST_QUESTIONS = [
    {
        "question": "How do agents communicate with each other in a web of agents?",
        "ground_truth": "Agents communicate through structured messaging protocols, handshaking mechanisms, and multi-layered architectures that enable agent-to-agent interaction and coordination.",
        "relevant_sources": ["Internet of Agents.pdf", "CAMEL.pdf", "L2M2 Multi-agent Coordination.pdf"]
    },
    {
        "question": "What is the role of an orchestrator agent in multi-agent systems?",
        "ground_truth": "An orchestrator agent manages and coordinates sub-agents by ranking and selecting them based on capability descriptions, delegating tasks, and performing continuous real-time evaluation.",
        "relevant_sources": ["L2M2 Multi-agent Coordination.pdf", "AgentVerse.pdf"]
    },
    {
        "question": "How is Salesforce monetizing AI agents?",
        "ground_truth": "Salesforce monetizes AI agents through three ways: upgrading existing seats to premium SKUs with embedded AI, new app deployments with higher ROI, and consumption-based flex credits for customer-facing agentic use cases.",
        "relevant_sources": ["Transcript-Salesforce-Inc-Q4-FY26-Earnings-Conference-Call-2-25-26.pdf"]
    },
    {
        "question": "What is Agentforce and how many deals has it closed?",
        "ground_truth": "Agentforce is Salesforce's multi-agent platform that closed 29,000 deals in its first 15 months, growing 50% quarter over quarter.",
        "relevant_sources": ["Transcript-Salesforce-Inc-Q4-FY26-Earnings-Conference-Call-2-25-26.pdf"]
    },

    # Semantic trap 1 — "web" means internet, not agent network
{
    "question": "How do I build a web scraper to collect agent data?",
    "ground_truth": "Not in corpus.",
    "relevant_sources": []
},

# Semantic trap 2 — "protocol" means communication etiquette, not agent protocol
{
    "question": "What communication protocol should I use for my API?",
    "ground_truth": "Not in corpus.",
    "relevant_sources": []
},

# Semantic trap 3 — "orchestration" means music, not agent orchestration  
{
    "question": "What makes a good musical orchestration?",
    "ground_truth": "Not in corpus.",
    "relevant_sources": []
},

# Semantic trap 4 — "agent" means real estate agent, not AI agent
{
    "question": "How do real estate agents find new clients?",
    "ground_truth": "Not in corpus.",
    "relevant_sources": []
},

# Semantic trap 5 — "token" means financial token, not LLM token
{
    "question": "Should I invest in crypto tokens in 2025?",
    "ground_truth": "Not in corpus.",
    "relevant_sources": []
},

# Adversarial question 1 — asks about something NOT in your corpus
{
    "question": "What is OpenAI's strategy for multi-agent systems?",
    "ground_truth": "Not covered in the document corpus.",
    "relevant_sources": []
},

# Adversarial question 2 — ambiguous, could match many documents
{
    "question": "What are agents?",
    "ground_truth": "Agents are autonomous systems that can perceive their environment and take actions.",
    "relevant_sources": ["Internet of Agents.pdf", "AgentVerse.pdf", "CAMEL.pdf"]
},

# Adversarial question 3 — requires synthesizing across documents
{
    "question": "How do academic research findings on agent coordination compare to how Salesforce implements it in Agentforce?",
    "ground_truth": "Academic research describes multi-layered coordination protocols and dynamic agent selection, while Salesforce implements this through Agentforce's orchestration layer with MCP servers and Slack integration.",
    "relevant_sources": ["L2M2 Multi-agent Coordination.pdf", "Transcript-Salesforce-Inc-Q4-FY26-Earnings-Conference-Call-2-25-26.pdf"]
},

# Adversarial question 4 — very specific number lookup
{
    "question": "How many Agentic Work Units did Salesforce deliver in Q4?",
    "ground_truth": "Salesforce delivered approximately 771 million Agentic Work Units in Q4 FY26.",
    "relevant_sources": ["Transcript-Salesforce-Inc-Q4-FY26-Earnings-Conference-Call-2-25-26.pdf"]
},

# Adversarial question 5 — cross-paper synthesis
{
    "question": "What are the key differences between CAMEL and AgentVerse approaches to multi-agent collaboration?",
    "ground_truth": "CAMEL focuses on communicative agents using role-playing for problem solving, while AgentVerse creates decentralized ecosystems where agents take specialized roles including recruiter, critic, and worker.",
    "relevant_sources": ["CAMEL.pdf", "AgentVerse.pdf"]
},

]

# ── Tier 1: Retrieval metrics (free, no LLM needed) ────────────────────

def compute_hit_rate(retrieved_docs, relevant_sources):
    if not relevant_sources:
        return 1.0 if not retrieved_docs else 0.0
    retrieved_sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
    for rel in relevant_sources:
        if any(rel in src for src in retrieved_sources):
            return 1.0
    return 0.0

def compute_mrr(retrieved_docs, relevant_sources):
    retrieved_sources = [doc.metadata.get("source", "") for doc in retrieved_docs]
    for rank, src in enumerate(retrieved_sources):
        if any(rel in src for rel in relevant_sources):
            return 1.0 / (rank + 1)
    return 0.0

def compute_precision_at_k(retrieved_docs, relevant_sources, k=3):
    if not relevant_sources:
        top_k = retrieved_docs[:k]
        return 0.0 if top_k else 1.0
    top_k = retrieved_docs[:k]
    hits = sum(
        1 for doc in top_k
        if any(rel in doc.metadata.get("source", "") for rel in relevant_sources)
    )
    return hits / k

def compute_recall_at_k(retrieved_docs, relevant_sources, k=3):
    if not relevant_sources:
        return 1.0 if not retrieved_docs else 0.0
    top_k = retrieved_docs[:k]
    retrieved_relevant = set(
        doc.metadata.get("source", "") for doc in top_k
        if any(rel in doc.metadata.get("source", "") for rel in relevant_sources)
    )
    return len(retrieved_relevant) / len(relevant_sources)

# ── Tier 2: Semantic similarity (free, embeddings only) ───────────────

def compute_semantic_similarity(answer, ground_truth, embedding_model):
    answer_vec = embedding_model.embed_query(answer)
    truth_vec = embedding_model.embed_query(ground_truth)
    score = cosine_similarity([answer_vec], [truth_vec])[0][0]
    return float(score)

# ── Tier 3: TruLens (LLM-as-judge, local Ollama) ──────────────────────

def run_trulens_evaluation(qa_pairs, reranked_docs_map):
    print("\n=== Running TruLens Evaluation ===")
    try:
        from trulens.core import TruSession
        from trulens.core import Feedback
        from trulens.providers.ollama import Ollama as TruOllama
        from trulens.apps.basic import TruBasicApp

        session = TruSession()
        session.reset_database()

        provider = TruOllama(model_engine="mistral")

        f_relevance = Feedback(
            provider.relevance_with_cot_reasons,
            name="Answer Relevance"
        ).on_input_output()

        f_context_relevance = Feedback(
            provider.context_relevance_with_cot_reasons,
            name="Context Relevance"
        ).on_input_output()

        def rag_app(question):
            return qa_pairs.get(question, "No answer available")

        tru_app = TruBasicApp(
            rag_app,
            app_name="AgentLens-RAG",
            feedbacks=[f_relevance, f_context_relevance]
        )

        for question in qa_pairs:
            with tru_app as recording:
                tru_app.app(question)

        leaderboard = session.get_leaderboard()
        print("\nTruLens Leaderboard:")
        print(leaderboard.to_string())
        leaderboard.to_csv("evaluation/trulens_results.csv", index=False)
        print("TruLens results saved to evaluation/trulens_results.csv")
        return leaderboard

    except Exception as e:
        print(f"TruLens evaluation failed: {e}")
        print("Tier 1 and Tier 2 results are still valid.")
        return None

# ── Main evaluation runner ─────────────────────────────────────────────

def run_evaluation():
    print("=== Loading pipeline components ===")
    vectorstore = load_vector_store()
    bm25, documents = build_bm25_index(vectorstore)
    reranker = load_reranker()
    llm = load_llm()
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    results = []
    qa_pairs = {}

    for item in TEST_QUESTIONS:
        query = item["question"]
        ground_truth = item["ground_truth"]
        relevant_sources = item["relevant_sources"]

        print(f"\nEvaluating: {query}")

        retrieved = hybrid_search(vectorstore, bm25, documents, query, k=20)
        reranked = rerank(reranker, query, retrieved, top_n=3)
        answer = generate_answer(llm, query, reranked)

        qa_pairs[query] = answer

        hit_rate = compute_hit_rate(reranked, relevant_sources)
        mrr = compute_mrr(reranked, relevant_sources)
        precision = compute_precision_at_k(reranked, relevant_sources, k=3)
        recall = compute_recall_at_k(reranked, relevant_sources, k=3)
        similarity = compute_semantic_similarity(
            answer, ground_truth, embedding_model
        )

        results.append({
            "question": query,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision@3": precision,
            "recall@3": recall,
            "semantic_similarity": similarity,
            "answer": answer[:200]
        })

        print(f"  Hit Rate:            {hit_rate:.4f}")
        print(f"  MRR:                 {mrr:.4f}")
        print(f"  Precision@3:         {precision:.4f}")
        print(f"  Recall@3:            {recall:.4f}")
        print(f"  Semantic Similarity: {similarity:.4f}")

    print("\n=== Overall Tier 1 + Tier 2 Results ===")
    print(f"  Avg Hit Rate:            {np.mean([r['hit_rate'] for r in results]):.4f}")
    print(f"  Avg MRR:                 {np.mean([r['mrr'] for r in results]):.4f}")
    print(f"  Avg Precision@3:         {np.mean([r['precision@3'] for r in results]):.4f}")
    print(f"  Avg Recall@3:            {np.mean([r['recall@3'] for r in results]):.4f}")
    print(f"  Avg Semantic Similarity: {np.mean([r['semantic_similarity'] for r in results]):.4f}")

    output_path = Path("evaluation/eval_results.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nTier 1 + 2 results saved to {output_path}")

    run_trulens_evaluation(qa_pairs, {})

if __name__ == "__main__":
    run_evaluation()