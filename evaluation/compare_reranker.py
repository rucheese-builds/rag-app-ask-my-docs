import sys
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import load_pipeline, run_pipeline
from evaluation.evaluate import (
    TEST_QUESTIONS, 
    compute_hit_rate, 
    compute_mrr, 
    compute_precision_at_k, 
    compute_recall_at_k, 
    compute_semantic_similarity
)
from langchain_ollama import OllamaEmbeddings

def evaluate_configuration(vectorstore, bm25_indices, reranker, llm, expansion_node, embedding_model, use_reranker):
    print(f"\n--- Running Evaluation with use_reranker={use_reranker} ---")
    results = []
    
    for i, item in enumerate(TEST_QUESTIONS):
        query = item["question"]
        ground_truth = item["ground_truth"]
        relevant_sources = item["relevant_sources"]
        
        start_time = time.time()
        res = run_pipeline(
            query, vectorstore, bm25_indices, reranker, llm, expansion_node,
            namespace="all", alpha=0.4, use_expansion=True, use_reranker=use_reranker
        )
        latency = time.time() - start_time
        
        answer = res["answer"]
        reranked = res.get("_reranked_docs", [])
        
        hit_rate = compute_hit_rate(reranked, relevant_sources)
        mrr = compute_mrr(reranked, relevant_sources)
        precision = compute_precision_at_k(reranked, relevant_sources, k=3)
        recall = compute_recall_at_k(reranked, relevant_sources, k=3)
        
        if res["in_domain"] and ground_truth != "Not in corpus.":
            try:
                similarity = compute_semantic_similarity(answer, ground_truth, embedding_model)
            except Exception:
                similarity = 0.0
        else:
            similarity = 1.0 if (not res["in_domain"] or ground_truth == "Not in corpus.") else 0.0
            
        results.append({
            "hit_rate": hit_rate,
            "mrr": mrr,
            "precision": precision,
            "recall": recall,
            "similarity": similarity,
            "latency": latency
        })
        print(f"Q{i+1:<2} | Hit Rate: {hit_rate:.2f} | Similarity: {similarity:.2f} | Latency: {latency:.2f}s")
        
    return {
        "hit_rate": np.mean([r["hit_rate"] for r in results]),
        "mrr": np.mean([r["mrr"] for r in results]),
        "precision": np.mean([r["precision"] for r in results]),
        "recall": np.mean([r["recall"] for r in results]),
        "similarity": np.mean([r["similarity"] for r in results]),
        "latency": np.mean([r["latency"] for r in results])
    }

def main():
    vectorstore, bm25_indices, reranker, llm, expansion_node = load_pipeline()
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Run with reranker
    metrics_with = evaluate_configuration(
        vectorstore, bm25_indices, reranker, llm, expansion_node, embedding_model, use_reranker=True
    )
    
    # Run without reranker
    metrics_without = evaluate_configuration(
        vectorstore, bm25_indices, None, llm, expansion_node, embedding_model, use_reranker=False
    )
    
    print("\n" + "="*60)
    print("           CROSS-ENCODER RERANKER COMPARISON SUMMARY")
    print("="*60)
    print(f"Metric              | With Reranker | Without Reranker | Delta")
    print(f"------------------------------------------------------------")
    for key in ["hit_rate", "mrr", "precision", "recall", "similarity", "latency"]:
        val_with = metrics_with[key]
        val_without = metrics_without[key]
        delta = val_with - val_without
        sign = "+" if delta >= 0 else ""
        unit = "s" if key == "latency" else ""
        print(f"{key:<19} | {val_with:.4f}{unit:<3} | {val_without:.4f}{unit:<10} | {sign}{delta:.4f}{unit}")
    print("="*60)

if __name__ == "__main__":
    main()
