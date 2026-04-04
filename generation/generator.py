from langchain_ollama import OllamaLLM
from langchain_core.documents import Document

LLM_MODEL = "llama3.2"

def load_llm():
    print(f"Loading LLM: {LLM_MODEL}")
    llm = OllamaLLM(model=LLM_MODEL)
    print("LLM loaded successfully")
    return llm

def format_context(documents):
    context_parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[Source {i+1}] — {source}\n{doc.page_content}"
        )
    return "\n\n".join(context_parts)

def build_prompt(query, context):
    return f"""You are an expert research assistant specializing in AI agents and multi-agent systems.

Answer the question using ONLY the information provided in the context below.
For every claim you make, you MUST cite the source using [Source N] notation.
If the answer is not found in the context, say "I don't have enough information in my documents to answer this."
Never fabricate information. Never use knowledge outside the provided context.

Context:
{context}

Question: {query}

Answer (with citations):"""

def generate_answer(llm, query, documents):
    print(f"\nGenerating answer for: {query}")
    context = format_context(documents)
    prompt = build_prompt(query, context)
    answer = llm.invoke(prompt)
    return answer

def format_response(query, answer, documents):
    sources_used = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        sources_used.append(f"  [Source {i+1}] {source} — page {page}")

    sources_section = "\n".join(sources_used)

    return f"""
=== Question ===
{query}

=== Answer ===
{answer}

=== Sources ===
{sources_section}
"""

if __name__ == "__main__":
    from retrieval.retriever import load_vector_store, build_bm25_index, hybrid_search
    from reranking.reranker import load_reranker, rerank

    vectorstore = load_vector_store()
    bm25, documents = build_bm25_index(vectorstore)
    reranker = load_reranker()
    llm = load_llm()

    test_query = "How do agents communicate with each other in a web of agents?"

    retrieved = hybrid_search(vectorstore, bm25, documents, test_query)
    reranked = rerank(reranker, test_query, retrieved, top_n=5)
    answer = generate_answer(llm, test_query, reranked)
    response = format_response(test_query, answer, reranked)

    print(response)