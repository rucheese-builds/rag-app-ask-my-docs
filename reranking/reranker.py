from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def load_reranker():
    print(f"Loading reranker model: {RERANKER_MODEL}")
    reranker = CrossEncoder(RERANKER_MODEL)
    print("Reranker loaded successfully")
    return reranker

def rerank(reranker, query, documents, top_n=3):
    print(f"\nReranking {len(documents)} chunks...")
    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(scores, documents))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    print(f"Top scores after cross-encoder:")
    for i, (score, doc) in enumerate(scored_docs[:5]):
        print(f"  [{i+1}] Score: {score:.4f} | Source: {doc.metadata.get('source', 'unknown')}")

    diverse_docs = diversity_rerank(scored_docs, top_n=top_n)
    print(f"After diversity reranking: {len(diverse_docs)} chunks from {len(set(d.metadata.get('source') for d in diverse_docs))} sources")
    return diverse_docs

def diversity_rerank(scored_docs, top_n=3):
    selected = []
    selected_sources = []

    for score, doc in scored_docs:
        source = doc.metadata.get('source', 'unknown')

        if len(selected) >= top_n:
            break

        if source not in selected_sources:
            selected.append(doc)
            selected_sources.append(source)
        elif selected_sources.count(source) < 2:
            selected.append(doc)
            selected_sources.append(source)

    if len(selected) < top_n:
        for score, doc in scored_docs:
            if doc not in selected:
                selected.append(doc)
            if len(selected) >= top_n:
                break

    return selected

if __name__ == "__main__":
    from retrieval.retriever import load_vector_store, build_bm25_index, hybrid_search

    vectorstore = load_vector_store()
    bm25, documents = build_bm25_index(vectorstore)

    test_query = "How do agents communicate with each other in a web of agents?"
    retrieved = hybrid_search(vectorstore, bm25, documents, test_query)

    reranker = load_reranker()
    final_chunks = rerank(reranker, test_query, retrieved, top_n=3)

    print("\n=== Final chunks after reranking ===")
    for i, doc in enumerate(final_chunks):
        print(f"\n[Chunk {i+1}] Source: {doc.metadata.get('source', 'unknown')}")
        print(doc.page_content[:300])