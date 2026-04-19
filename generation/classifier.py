from langchain_ollama import OllamaLLM

def classify_query(query, llm=None):
    classifier_llm = OllamaLLM(model="mistral")

    prompt = f"""Is this question about AI agents, multi-agent systems, agent frameworks, or enterprise AI strategy?

Question: {query}

Answer yes or no only."""

    result = classifier_llm.invoke(prompt).strip().lower()
    is_in_domain = result.startswith("yes")
    print(f"Query classification: {'IN_DOMAIN' if is_in_domain else 'OUT_OF_DOMAIN'} ({result[:20]})")
    return is_in_domain