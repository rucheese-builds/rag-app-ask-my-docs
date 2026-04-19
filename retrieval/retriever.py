from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import pickle
import os

CHROMA_DIR = Path("chroma_db")
BM25_INDEX_PATH = Path("bm25_index.pkl")

def load_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    print(f"Vector store loaded with {vectorstore._collection.count()} chunks")
    return vectorstore

def build_bm25_index(vectorstore):
    if BM25_INDEX_PATH.exists():
        print("Loading existing BM25 index...")
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        return data["bm25"], data["documents"]

    print("Building BM25 index...")
    results = vectorstore._collection.get()
    documents = []
    for i, doc_text in enumerate(results["documents"]):
        metadata = results["metadatas"][i] if results["metadatas"] else {}
        documents.append(Document(page_content=doc_text, metadata=metadata))

    tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": documents}, f)

    print(f"BM25 index built over {len(documents)} chunks")
    return bm25, documents

def vector_search(vectorstore, query, k=20):
    results = vectorstore.similarity_search(query, k=k)
    return results

def bm25_search(bm25, documents, query, k=20):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)),
                        key=lambda i: scores[i],
                        reverse=True)[:k]
    return [documents[i] for i in top_indices]

def reciprocal_rank_fusion(bm25_results, vector_results, k=60):
    scores = {}
    doc_map = {}

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    for rank, doc in enumerate(vector_results):
        key = doc.page_content[:100]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        doc_map[key] = doc

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[key] for key in sorted_keys]

def hybrid_search(vectorstore, bm25, documents, query, k=100):
    print(f"\nSearching for: {query}")
    vector_results = vector_search(vectorstore, query, k=k)
    print(f"Vector search returned {len(vector_results)} chunks")
    bm25_results = bm25_search(bm25, documents, query, k=k)
    print(f"BM25 search returned {len(bm25_results)} chunks")
    fused_results = reciprocal_rank_fusion(bm25_results, vector_results)
    top_results = fused_results[:25]
    print(f"After fusion: top {len(top_results)} chunks selected for reranking")
    return top_results

if __name__ == "__main__":
    vectorstore = load_vector_store()
    bm25, documents = build_bm25_index(vectorstore)

    test_query = "How do agents communicate with each other in a web of agents?"
    results = hybrid_search(vectorstore, bm25, documents, test_query)

    print("\n=== Top retrieved chunks ===")
    for i, doc in enumerate(results):
        print(f"\n[Chunk {i+1}] Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content[:300])