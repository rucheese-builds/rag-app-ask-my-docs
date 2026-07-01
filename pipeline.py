from generation.classifier import classify_query
from generation.expansion import QueryExpansionNode
from retrieval.retriever import load_vector_store, load_bm25_indices, hybrid_search
from reranking.reranker import load_reranker, rerank
from generation.generator import load_llm, generate_answer

OUT_OF_DOMAIN_RESPONSE = """This question appears to be outside the scope
of my document corpus, which covers:

- Web of Agents research papers
- Multi-agent system architectures
- Enterprise AI adoption (Salesforce, Microsoft, Nvidia, ServiceNow, Google, IBM)

Please ask something related to these topics."""

def load_pipeline():
    print("[Pipeline] Loading pipeline components...")
    vectorstore = load_vector_store()
    bm25_indices = load_bm25_indices()
    reranker = load_reranker()
    llm = load_llm()
    expansion_node = QueryExpansionNode()
    print("[Pipeline] Pipeline ready.")
    return vectorstore, bm25_indices, reranker, llm, expansion_node

def run_pipeline(query, vectorstore, bm25_indices, reranker, llm, expansion_node, 
                 namespace="all", alpha=0.5, use_expansion=True, use_reranker=True):
    
    # 1. Classify Query
    is_in_domain = classify_query(query, llm)
    if not is_in_domain:
        return {
            "answer": OUT_OF_DOMAIN_RESPONSE,
            "sources": [],
            "in_domain": False,
            "query": query,
            "expanded_query": query,
            "_reranked_docs": []
        }

    # 2. Query Expansion (if enabled)
    expanded_query = query
    if use_expansion and expansion_node:
        expanded_query = expansion_node.expand(query)

    # 3. Hybrid Search (Convex Score Combination)
    retrieved_child_chunks = hybrid_search(
        vectorstore, 
        bm25_indices, 
        expanded_query, 
        namespace=namespace, 
        alpha=alpha, 
        k=20
    )

    if not retrieved_child_chunks:
        return {
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": [],
            "in_domain": True,
            "query": query,
            "expanded_query": expanded_query,
            "_reranked_docs": []
        }

    # 4. Rerank Child Chunks (or bypass)
    if use_reranker and reranker:
        reranked_child_chunks = rerank(reranker, expanded_query, retrieved_child_chunks, top_n=3)
    else:
        print("[Pipeline] Reranker bypassed, taking top 3 from Convex Score Combination.")
        reranked_child_chunks = retrieved_child_chunks[:3]

    # 5. Swap Child Content with Parent Content
    # We pass the parent content to the generator for complete context
    generation_chunks = []
    for doc in reranked_child_chunks:
        parent_content = doc.metadata.get("parent_content")
        if parent_content:
            # Create a new document to avoid modifying in-place permanently or just modify copy
            parent_doc = doc.__class__(
                page_content=parent_content,
                metadata=doc.metadata
            )
            generation_chunks.append(parent_doc)
        else:
            generation_chunks.append(doc)

    # 6. Generate Answer using parent context
    answer = generate_answer(llm, query, generation_chunks)

    # 7. Collect sources
    sources = []
    for i, doc in enumerate(reranked_child_chunks):
        sources.append({
            "index": i + 1,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "namespace": doc.metadata.get("namespace", "unknown"),
            "content": doc.page_content, # original child content for display
            "parent_content": doc.metadata.get("parent_content", doc.page_content)[:300]
        })

    return {
        "answer": answer,
        "sources": sources,
        "in_domain": True,
        "query": query,
        "expanded_query": expanded_query,
        "_reranked_docs": reranked_child_chunks
    }
def load_neo4j_pipeline():
    import os
    from llama_index.core import PropertyGraphIndex
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai import OpenAI

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD is not set in your .env file!")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in your .env file!")

    print("[Pipeline] Connecting to Neo4j Property Graph (Read-Only)...")
    graph_store = Neo4jPropertyGraphStore(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        database=neo4j_database
    )

    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=HuggingFaceEmbedding(
            model_name=os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
            cache_folder="./hf_cache"
        ),
        llm=OpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    )
    
    print("[Pipeline] Neo4j read-only query engine loaded successfully.")
    return index.as_query_engine()

def run_neo4j_pipeline(query, query_engine):
    response = query_engine.query(query)
    
    sources = []
    for node_with_score in response.source_nodes:
        node = node_with_score.node
        text = node.get_content()
        meta = node.metadata
        
        sources.append({
            "index": len(sources) + 1,
            "source": meta.get("source", "Unknown Document"),
            "page": meta.get("page", 1),
            "namespace": meta.get("namespace", "all"),
            "content": text[:400],
            "parent_content": text[:1500]
        })
        
    return {
        "answer": str(response.response),
        "sources": sources,
        "in_domain": True,
        "query": query,
        "expanded_query": query,
        "_reranked_docs": []
    }

if __name__ == "__main__":
    vectorstore, bm25_indices, reranker, llm, expansion_node = load_pipeline()

    test_queries = [
        "What is CAMEL?",
        "How is Salesforce monetizing Agentforce?",
        "Should I invest in crypto?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = run_pipeline(
            query, vectorstore, bm25_indices, reranker, llm, expansion_node,
            namespace="all", alpha=0.4, use_expansion=True
        )
        print(f"In domain: {result['in_domain']}")
        print(f"Expanded Query: {result.get('expanded_query')}")
        print(f"Answer: {result['answer']}")
        if result['sources']:
            print("Sources:")
            for s in result['sources']:
                print(f"  [{s['index']}] {s['source']} (page {s['page']}) in namespace '{s['namespace']}'")