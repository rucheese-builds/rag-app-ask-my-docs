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
    print("=== Loading RAG pipelines (Vector + Graph) & evaluation components ===")
    
    # Load Vector RAG pipeline
    vectorstore, bm25_indices, reranker, llm, expansion_node = load_pipeline()
    
    # Load Graph RAG pipeline
    try:
        from pipeline import load_neo4j_pipeline, run_neo4j_pipeline
        neo4j_query_engine = load_neo4j_pipeline()
    except Exception as e:
        print(f"[Warning] Failed to load Neo4j Property Graph pipeline: {e}")
        neo4j_query_engine = None

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

    vector_results = []
    graph_results = []
    
    for idx, item in enumerate(records):
        query = item["question"]
        ground_truth = item["ground_truth"]
        keywords_str = item["mandatory_keywords"]
        category = item["category"]
        
        # Sources are comma-separated in the CSV
        sources_str = item.get("relevant_sources", "")
        relevant_sources = [s.strip() for s in sources_str.split(",") if s.strip()]

        print(f"\n[{idx+1}/{len(records)}] Evaluating Category: {category} | Query: '{query}'")

        # ----------------- VECTOR PIPELINE RUN -----------------
        print("  --> Running Vector RAG Pipeline...")
        res_v = run_pipeline(
            query, vectorstore, bm25_indices, reranker, llm, expansion_node,
            namespace="all", alpha=0.4, use_expansion=True
        )
        ans_v = res_v["answer"]
        reranked_docs_v = res_v.get("_reranked_docs", [])
        is_out_v = not res_v.get("in_domain", True)

        hit_v = compute_hit_rate(reranked_docs_v, relevant_sources)
        mrr_v = compute_mrr(reranked_docs_v, relevant_sources)
        sim_v = compute_semantic_similarity(ans_v, ground_truth, embedding_model)
        key_v = compute_keyword_coverage(ans_v, keywords_str, is_out_v or category == "Negative" or "outside the scope" in ground_truth.lower())

        context_v = "\n\n".join([doc.page_content for doc in reranked_docs_v])
        
        print("      Running Vector LLM-as-a-judge metrics...")
        faith_v = evaluate_faithfulness(eval_llm, query, context_v, ans_v)
        rel_v = evaluate_answer_relevance(eval_llm, query, ans_v)
        prec_v = evaluate_context_precision(eval_llm, query, reranked_docs_v)
        rec_v = evaluate_context_recall(eval_llm, ground_truth, context_v)

        vector_results.append({
            "question": query, "category": category, "hit_rate": hit_v, "mrr": mrr_v,
            "semantic_similarity": sim_v, "keyword_coverage": key_v, "faithfulness": faith_v,
            "answer_relevance": rel_v, "context_precision": prec_v, "context_recall": rec_v,
            "generated_answer": ans_v
        })

        # ----------------- GRAPH PIPELINE RUN -----------------
        if neo4j_query_engine:
            print("  --> Running Graph RAG Pipeline...")
            res_g = run_neo4j_pipeline(query, neo4j_query_engine)
            ans_g = res_g["answer"]
            
            # Map Neo4j source nodes to compatible document structures for metrics
            sources_g = res_g.get("sources", [])
            class MockDoc:
                def __init__(self, metadata, page_content):
                    self.metadata = metadata
                    self.page_content = page_content
            docs_g = [MockDoc({"source": s["source"], "page": s["page"], "namespace": s["namespace"]}, s["content"]) for s in sources_g]
            
            hit_g = compute_hit_rate(docs_g, relevant_sources)
            mrr_g = compute_mrr(docs_g, relevant_sources)
            sim_g = compute_semantic_similarity(ans_g, ground_truth, embedding_model)
            key_g = compute_keyword_coverage(ans_g, keywords_str, category == "Negative" or "outside the scope" in ground_truth.lower())

            context_g = "\n\n".join([s["content"] for s in sources_g])
            
            print("      Running Graph LLM-as-a-judge metrics...")
            faith_g = evaluate_faithfulness(eval_llm, query, context_g, ans_g)
            rel_g = evaluate_answer_relevance(eval_llm, query, ans_g)
            prec_g = evaluate_context_precision(eval_llm, query, docs_g)
            rec_g = evaluate_context_recall(eval_llm, ground_truth, context_g)

            graph_results.append({
                "question": query, "category": category, "hit_rate": hit_g, "mrr": mrr_g,
                "semantic_similarity": sim_g, "keyword_coverage": key_g, "faithfulness": faith_g,
                "answer_relevance": rel_g, "context_precision": prec_g, "context_recall": rec_g,
                "generated_answer": ans_g
            })
        else:
            graph_results.append({
                "question": query, "category": category, "hit_rate": 0.0, "mrr": 0.0,
                "semantic_similarity": 0.0, "keyword_coverage": 0.0, "faithfulness": 0.0,
                "answer_relevance": 0.0, "context_precision": 0.0, "context_recall": 0.0,
                "generated_answer": "Neo4j Query Engine Bypassed."
            })

    # Save detailed evaluation outputs to CSV
    COMP_RESULTS_PATH = Path("evaluation/eval_results_comparative.csv")
    with open(COMP_RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["question", "category", "pipeline", "hit_rate", "mrr", "semantic_similarity", "keyword_coverage", "faithfulness", "answer_relevance", "context_precision", "context_recall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for vr, gr in zip(vector_results, graph_results):
            vr_row = {k: vr[k] for k in fieldnames if k != "pipeline"}
            vr_row["pipeline"] = "Vector RAG"
            writer.writerow(vr_row)
            
            gr_row = {k: gr[k] for k in fieldnames if k != "pipeline"}
            gr_row["pipeline"] = "Graph RAG"
            writer.writerow(gr_row)
            
    print(f"\nComparative evaluation results saved to {COMP_RESULTS_PATH}")

    # Averages summary
    print("\n" + "="*95)
    print("📊 SIDE-BY-SIDE RAG COMPARATIVE METRICS SUMMARY (Vector RAG vs. Graph RAG)")
    print("="*95)
    print(f"{'Pipeline':<12} | {'HitRate':<7} | {'MRR':<6} | {'SemSim':<6} | {'KeyCov':<6} | {'Faith':<5} | {'AnsRel':<6} | {'CtxPrec':<7} | {'CtxRec':<6}")
    print("-"*95)
    
    # Vector averages
    v_hit = np.mean([r["hit_rate"] for r in vector_results])
    v_mrr = np.mean([r["mrr"] for r in vector_results])
    v_sim = np.mean([r["semantic_similarity"] for r in vector_results])
    v_key = np.mean([r["keyword_coverage"] for r in vector_results])
    v_faith = np.mean([r["faithfulness"] for r in vector_results])
    v_rel = np.mean([r["answer_relevance"] for r in vector_results])
    v_prec = np.mean([r["context_precision"] for r in vector_results])
    v_rec = np.mean([r["context_recall"] for r in vector_results])
    print(f"{'Vector RAG':<12} | {v_hit:.4f} | {v_mrr:.4f} | {v_sim:.4f} | {v_key:.4f} | {v_faith:.4f} | {v_rel:.4f} | {v_prec:.4f} | {v_rec:.4f}")
    
    # Graph averages
    g_hit = np.mean([r["hit_rate"] for r in graph_results])
    g_mrr = np.mean([r["mrr"] for r in graph_results])
    g_sim = np.mean([r["semantic_similarity"] for r in graph_results])
    g_key = np.mean([r["keyword_coverage"] for r in graph_results])
    g_faith = np.mean([r["faithfulness"] for r in graph_results])
    g_rel = np.mean([r["answer_relevance"] for r in graph_results])
    g_prec = np.mean([r["context_precision"] for r in graph_results])
    g_rec = np.mean([r["context_recall"] for r in graph_results])
    print(f"{'Graph RAG':<12} | {g_hit:.4f} | {g_mrr:.4f} | {g_sim:.4f} | {g_key:.4f} | {g_faith:.4f} | {g_rel:.4f} | {g_prec:.4f} | {g_rec:.4f}")
    print("="*95)

if __name__ == "__main__":
    run_evaluation()