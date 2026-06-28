import sys
import numpy as np
import csv
import re
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings
from pipeline import load_pipeline, run_pipeline, OUT_OF_DOMAIN_RESPONSE
from evaluation.metrics import (
    get_eval_llm,
    evaluate_faithfulness,
    evaluate_answer_relevance,
    evaluate_context_precision,
    evaluate_context_recall
)

CSV_PATH = Path("evaluation/golden_dataset.csv")
EVAL_RESULTS_PATH = Path("evaluation/eval_results_golden.csv")

def compute_hit_rate(retrieved_docs, relevant_sources_list):
    if not relevant_sources_list:
        return 1.0 if not retrieved_docs else 0.0
    retrieved_sources = [doc.metadata.get("source", "").lower() for doc in retrieved_docs]
    for rel in relevant_sources_list:
        if any(rel.lower() in src for src in retrieved_sources):
            return 1.0
    return 0.0

def compute_mrr(retrieved_docs, relevant_sources_list):
    if not relevant_sources_list:
        return 1.0 if not retrieved_docs else 0.0
    retrieved_sources = [doc.metadata.get("source", "").lower() for doc in retrieved_docs]
    for rank, src in enumerate(retrieved_sources):
        if any(rel.lower() in src for rel in relevant_sources_list):
            return 1.0 / (rank + 1)
    return 0.0

def compute_precision_at_k(retrieved_docs, relevant_sources_list, k=3):
    if not relevant_sources_list:
        return 0.0 if retrieved_docs[:k] else 1.0
    top_k = retrieved_docs[:k]
    hits = sum(
        1 for doc in top_k
        if any(rel.lower() in doc.metadata.get("source", "").lower() for rel in relevant_sources_list)
    )
    return hits / k

def compute_recall_at_k(retrieved_docs, relevant_sources_list, k=3):
    if not relevant_sources_list:
        return 1.0 if not retrieved_docs else 0.0
    top_k = retrieved_docs[:k]
    retrieved_relevant = set(
        doc.metadata.get("source", "").lower() for doc in top_k
        if any(rel.lower() in doc.metadata.get("source", "").lower() for rel in relevant_sources_list)
    )
    return len(retrieved_relevant) / len(relevant_sources_list)

def compute_semantic_similarity(answer, ground_truth, embedding_model):
    if "outside the scope" in ground_truth.lower() and "outside the scope" in answer.lower():
        return 1.0
    try:
        answer_vec = embedding_model.embed_query(answer)
        truth_vec = embedding_model.embed_query(ground_truth)
        score = cosine_similarity([answer_vec], [truth_vec])[0][0]
        return float(score)
    except Exception as e:
        print(f"Embedding similarity failed: {e}")
        return 0.0

def compute_keyword_coverage(answer, mandatory_keywords_str, is_out_of_domain):
    """
    Checks if mandatory keywords appear in the answer (case-insensitive substring check).
    For out-of-domain answers, if the model returns the out-of-domain response, score is 1.0, else 0.0.
    """
    if is_out_of_domain:
        # Check if the answer indicates it is outside the scope or doesn't have info
        is_correct_reject = (
            "outside the scope" in answer.lower() or 
            "don't have enough information" in answer.lower() or
            "do not have enough information" in answer.lower()
        )
        return 1.0 if is_correct_reject else 0.0

    if not mandatory_keywords_str:
        return 1.0

    keywords = [k.strip().lower() for k in mandatory_keywords_str.split(",") if k.strip()]
    if not keywords:
        return 1.0

    matched = 0
    ans_lower = answer.lower()
    for kw in keywords:
        # simple substring search
        if kw in ans_lower:
            matched += 1
            
    return matched / len(keywords)

def run_evaluation():
    print("=== Loading RAG pipeline & evaluation components ===")
    vectorstore, bm25_indices, reranker, llm, expansion_node = load_pipeline()
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    eval_llm = get_eval_llm()

    if not CSV_PATH.exists():
        print(f"Error: Golden dataset not found at {CSV_PATH}. Please run seed_dataset.py first.")
        return

    # Load records from Golden Dataset
    records = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Loaded {len(records)} test questions from {CSV_PATH}")

    results = []
    
    for idx, item in enumerate(records):
        query = item["question"]
        ground_truth = item["ground_truth"]
        keywords_str = item["mandatory_keywords"]
        category = item["category"]
        
        # Sources are comma-separated in the CSV
        sources_str = item.get("relevant_sources", "")
        relevant_sources = [s.strip() for s in sources_str.split(",") if s.strip()]

        print(f"\n[{idx+1}/{len(records)}] Evaluating ({category}): '{query}'")

        # Run pipeline
        res = run_pipeline(
            query, vectorstore, bm25_indices, reranker, llm, expansion_node,
            namespace="all", alpha=0.4, use_expansion=True
        )
        answer = res["answer"]
        reranked_docs = res.get("_reranked_docs", [])
        is_out_of_domain = not res.get("in_domain", True)

        # 1. Retrieval Metrics
        hit_rate = compute_hit_rate(reranked_docs, relevant_sources)
        mrr = compute_mrr(reranked_docs, relevant_sources)
        precision = compute_precision_at_k(reranked_docs, relevant_sources, k=3)
        recall = compute_recall_at_k(reranked_docs, relevant_sources, k=3)

        # 2. Semantic Similarity
        similarity = compute_semantic_similarity(answer, ground_truth, embedding_model)

        # 3. Keyword Coverage (Fact Recall)
        keyword_coverage = compute_keyword_coverage(answer, keywords_str, is_out_of_domain or category == "Negative" or "outside the scope" in ground_truth.lower())

        # 4. Advanced LLM-as-a-judge metrics
        context_text = "\n\n".join([doc.page_content for doc in reranked_docs])
        
        print("  Running LLM-as-a-judge metrics (Faithfulness, Answer Relevance, Context Precision/Recall)...")
        faithfulness = evaluate_faithfulness(eval_llm, query, context_text, answer)
        answer_relevance = evaluate_answer_relevance(eval_llm, query, answer)
        context_precision = evaluate_context_precision(eval_llm, query, reranked_docs)
        context_recall = evaluate_context_recall(eval_llm, ground_truth, context_text)

        print(f"  Hit Rate:          {hit_rate:.4f} | MRR: {mrr:.4f}")
        print(f"  Semantic Sim:      {similarity:.4f} | Keyword Coverage: {keyword_coverage:.4f}")
        print(f"  Faithfulness:      {faithfulness:.4f} | Answer Relevance: {answer_relevance:.4f}")
        print(f"  Context Precision: {context_precision:.4f} | Context Recall: {context_recall:.4f}")

        results.append({
            "question": query,
            "category": category,
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision_at_3": precision,
            "recall_at_3": recall,
            "semantic_similarity": similarity,
            "keyword_coverage": keyword_coverage,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "generated_answer": answer.replace("\n", " ")[:150]
        })

    # Save detailed evaluation outputs
    with open(EVAL_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed evaluation results saved to {EVAL_RESULTS_PATH}")

    # Sliced metrics calculation
    categories = sorted(list(set(r["category"] for r in results)))
    
    print("\n" + "="*80)
    print("📊 CATEGORY-SLICED METRICS SUMMARY")
    print("="*80)
    print(f"{'Category':<22} | {'Count':<5} | {'HitRate':<7} | {'Recall':<6} | {'SemSim':<6} | {'KeyCov':<6} | {'Faith':<5} | {'AnsRel':<6} | {'CtxPrec':<7} | {'CtxRec':<6}")
    print("-"*106)
    
    for cat in categories:
        cat_rows = [r for r in results if r["category"] == cat]
        count = len(cat_rows)
        avg_hit = np.mean([r["hit_rate"] for r in cat_rows])
        avg_rec = np.mean([r["recall_at_3"] for r in cat_rows])
        avg_sim = np.mean([r["semantic_similarity"] for r in cat_rows])
        avg_key = np.mean([r["keyword_coverage"] for r in cat_rows])
        avg_faith = np.mean([r["faithfulness"] for r in cat_rows])
        avg_rel = np.mean([r["answer_relevance"] for r in cat_rows])
        avg_cprec = np.mean([r["context_precision"] for r in cat_rows])
        avg_crec = np.mean([r["context_recall"] for r in cat_rows])
        
        print(f"{cat:<22} | {count:<5} | {avg_hit:.4f} | {avg_rec:.4f} | {avg_sim:.4f} | {avg_key:.4f} | {avg_faith:.4f} | {avg_rel:.4f} | {avg_cprec:.4f} | {avg_crec:.4f}")

    print("-"*106)
    # Overall averages
    avg_hit = np.mean([r["hit_rate"] for r in results])
    avg_rec = np.mean([r["recall_at_3"] for r in results])
    avg_sim = np.mean([r["semantic_similarity"] for r in results])
    avg_key = np.mean([r["keyword_coverage"] for r in results])
    avg_faith = np.mean([r["faithfulness"] for r in results])
    avg_rel = np.mean([r["answer_relevance"] for r in results])
    avg_cprec = np.mean([r["context_precision"] for r in results])
    avg_crec = np.mean([r["context_recall"] for r in results])
    
    print(f"{'OVERALL AVERAGE':<22} | {len(results):<5} | {avg_hit:.4f} | {avg_rec:.4f} | {avg_sim:.4f} | {avg_key:.4f} | {avg_faith:.4f} | {avg_rel:.4f} | {avg_cprec:.4f} | {avg_crec:.4f}")
    print("="*80)

if __name__ == "__main__":
    run_evaluation()