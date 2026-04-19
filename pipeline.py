from generation.classifier import classify_query
from retrieval.retriever import load_vector_store, build_bm25_index, hybrid_search
from reranking.reranker import load_reranker, rerank
from generation.generator import load_llm, generate_answer

OUT_OF_DOMAIN_RESPONSE = """This question appears to be outside the scope
of my document corpus, which covers:

- Web of Agents research papers
- Multi-agent system architectures
- Enterprise AI adoption (Salesforce, Microsoft, Nvidia, ServiceNow, Google, IBM)

Please ask something related to these topics."""

def load_pipeline():
    print("Loading pipeline components...")
    vectorstore = load_vector_store()
    bm25, documents = build_bm25_index(vectorstore)
    reranker = load_reranker()
    llm = load_llm()
    print("Pipeline ready.")
    return vectorstore, bm25, documents, reranker, llm

def run_pipeline(query, vectorstore, bm25, documents, reranker, llm):
    is_in_domain = classify_query(query, llm)

    if not is_in_domain:
        return {
            "answer": OUT_OF_DOMAIN_RESPONSE,
            "sources": [],
            "in_domain": False,
            "query": query
        }

    retrieved = hybrid_search(vectorstore, bm25, documents, query)
    reranked = rerank(reranker, query, retrieved, top_n=2)
    answer = generate_answer(llm, query, reranked)

    sources = []
    for i, doc in enumerate(reranked):
        sources.append({
            "index": i + 1,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "content": doc.page_content[:300]
        })

    return {
        "answer": answer,
        "sources": sources,
        "in_domain": True,
        "query": query
    }

if __name__ == "__main__":
    vectorstore, bm25, documents, reranker, llm = load_pipeline()

    test_queries = [
        "How do agents communicate in a web of agents?",
        "How do I build a web scraper to collect agent data?",
        "What is Salesforce Agentforce?",
        "Should I invest in crypto tokens?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = run_pipeline(
            query, vectorstore, bm25, documents, reranker, llm
        )
        print(f"In domain: {result['in_domain']}")
        print(f"Answer: {result['answer'][:300]}")
        if result['sources']:
            print("Sources:")
            for s in result['sources']:
                print(f"  [{s['index']}] {s['source']} — page {s['page']}")