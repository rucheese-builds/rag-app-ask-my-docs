from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle
import os

CHROMA_DIR = Path("chroma_db")
BM25_RESEARCH_PATH = Path("bm25_research.pkl")
BM25_EARNINGS_PATH = Path("bm25_earnings.pkl")

def load_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    print(f"[Retriever] Vector store loaded with {vectorstore._collection.count()} child chunks.")
    return vectorstore

def load_bm25_indices():
    indices = {}
    
    if BM25_RESEARCH_PATH.exists():
        print("[Retriever] Loading research BM25 index...")
        with open(BM25_RESEARCH_PATH, "rb") as f:
            data = pickle.load(f)
        indices["research"] = (data["bm25"], data["documents"])
    else:
        print("[Retriever] WARNING: Research BM25 index file not found.")
        
    if BM25_EARNINGS_PATH.exists():
        print("[Retriever] Loading earnings BM25 index...")
        with open(BM25_EARNINGS_PATH, "rb") as f:
            data = pickle.load(f)
        indices["earning-reports"] = (data["bm25"], data["documents"])
    else:
        print("[Retriever] WARNING: Earnings BM25 index file not found.")
        
    return indices

def vector_search_with_scores(vectorstore, query, namespace="all", k=20):
    """Run vector search filtered by namespace and return documents with distances."""
    filter_dict = None
    if namespace != "all":
        filter_dict = {"namespace": namespace}
        
    # similarity_search_with_score returns list of (Document, L2 distance/cosine distance)
    results = vectorstore.similarity_search_with_score(
        query, 
        k=k,
        filter=filter_dict
    )
    return results

def bm25_search_with_scores(bm25, documents, query, k=20):
    """Run BM25 search over documents and return documents with raw scores."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    scored_docs = list(zip(documents, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_docs[:k]

def normalize_dense_results(results):
    """Normalize dense distances where smaller distance is better (best -> 1.0, worst -> 0.0)."""
    if not results:
        return []
    dists = [r[1] for r in results]
    min_dist = min(dists)
    max_dist = max(dists)
    
    norm_results = []
    for doc, dist in results:
        if max_dist != min_dist:
            norm_score = 1.0 - (dist - min_dist) / (max_dist - min_dist)
        else:
            norm_score = 1.0
        norm_results.append((doc, norm_score))
    return norm_results

def normalize_sparse_results(results):
    """Normalize sparse scores where larger score is better (best -> 1.0, worst -> 0.0)."""
    if not results:
        return []
    scores = [r[1] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    
    norm_results = []
    for doc, score in results:
        if max_score != min_score:
            norm_score = (score - min_score) / (max_score - min_score)
        else:
            norm_score = 1.0
        norm_results.append((doc, norm_score))
    return norm_results

def convex_fusion(dense_norm, sparse_norm, alpha=0.5):
    """Perform Convex Combination of dense and sparse scores: alpha * dense + (1 - alpha) * sparse."""
    fused = {}
    
    # Add dense results
    for doc, score in dense_norm:
        key = doc.page_content
        if key not in fused:
            fused[key] = {"doc": doc, "dense": score, "sparse": 0.0}
        else:
            fused[key]["dense"] = score
            
    # Add sparse results
    for doc, score in sparse_norm:
        key = doc.page_content
        if key not in fused:
            fused[key] = {"doc": doc, "dense": 0.0, "sparse": score}
        else:
            fused[key]["sparse"] = score
            
    # Compute convex score
    scored_fused_docs = []
    for key, data in fused.items():
        doc = data["doc"]
        final_score = alpha * data["dense"] + (1.0 - alpha) * data["sparse"]
        scored_fused_docs.append((doc, final_score))
        
    # Sort descending
    scored_fused_docs.sort(key=lambda x: x[1], reverse=True)
    return scored_fused_docs

def hybrid_search(vectorstore, bm25_indices, query, namespace="all", alpha=0.5, k=100):
    print(f"\n[Retriever] Hybrid Search for: '{query}' | Namespace: {namespace} | Alpha: {alpha}")
    
    # 1. Vector Search
    vector_results = vector_search_with_scores(vectorstore, query, namespace, k=k)
    print(f"[Retriever] Vector search returned {len(vector_results)} chunks.")
    
    # 2. BM25 Search
    bm25_results = []
    if namespace == "all":
        # Search both research and earnings and merge
        res_results = []
        if "research" in bm25_indices:
            res_bm25, res_docs = bm25_indices["research"]
            res_results = bm25_search_with_scores(res_bm25, res_docs, query, k=k)
        earn_results = []
        if "earning-reports" in bm25_indices:
            earn_bm25, earn_docs = bm25_indices["earning-reports"]
            earn_results = bm25_search_with_scores(earn_bm25, earn_docs, query, k=k)
        # Combine
        bm25_results = res_results + earn_results
    else:
        if namespace in bm25_indices:
            bm25, docs = bm25_indices[namespace]
            bm25_results = bm25_search_with_scores(bm25, docs, query, k=k)
            
    print(f"[Retriever] BM25 search returned {len(bm25_results)} chunks.")
    
    # 3. Normalisation
    dense_norm = normalize_dense_results(vector_results)
    sparse_norm = normalize_sparse_results(bm25_results)
    
    # 4. Fusion
    fused_with_scores = convex_fusion(dense_norm, sparse_norm, alpha=alpha)
    top_results = [doc for doc, score in fused_with_scores[:25]]
    
    print(f"[Retriever] After Convex combination: top {len(top_results)} chunks selected for reranking.")
    return top_results

if __name__ == "__main__":
    vectorstore = load_vector_store()
    bm25_indices = load_bm25_indices()
    
    test_query = "What is CAMEL multi-agent?"
    results = hybrid_search(vectorstore, bm25_indices, test_query, namespace="research", alpha=0.4)
    
    print("\n=== Top retrieved chunks ===")
    for i, doc in enumerate(results[:3]):
        print(f"\n[Chunk {i+1}] Source: {doc.metadata.get('source')} | Namespace: {doc.metadata.get('namespace')}")
        print(doc.page_content[:200])