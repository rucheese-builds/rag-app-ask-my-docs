from langchain_ollama import OllamaLLM

def classify_query(query, llm=None):
    # In-domain keyword checks (instant, zero latency, deterministic)
    in_domain_keywords = [
        "camel", "autogen", "agentverse", "react", "dylan", "l2m2", 
        "agentbench", "openagents", "agentforce", "agentic", "salesforce", 
        "servicenow", "nvidia", "microsoft", "ibm", "google", "earnings", 
        "transcript", "web of agents", "multi-agent", "agent coordination",
        "agentrank", "dovis", "internet 3.0", "internet of agents"
    ]
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in in_domain_keywords):
        print(f"Query classification: IN_DOMAIN (keyword match)")
        return True

    from langchain_ollama import OllamaLLM
    classifier_llm = OllamaLLM(model="mistral")

    prompt = f"""You are a query classifier for a RAG system. The document corpus covers:
1. Academic research papers on AI agents, multi-agent frameworks (CAMEL, AutoGen, AgentVerse, L2M2, ReAct, DyLAN, AgentRank, DOVIS), agent benchmarks, and the Internet of Agents / Internet 3.0.
2. Enterprise AI strategies and business adoption (Salesforce, Microsoft, Nvidia, ServiceNow, Google, IBM earnings calls).

Is the following question related to these topics? Answer with 'yes' or 'no' only.

Question: {query}

Answer:"""

    try:
        result = classifier_llm.invoke(prompt).strip().lower()
        is_in_domain = "yes" in result or result.startswith("yes")
        print(f"Query classification: {'IN_DOMAIN' if is_in_domain else 'OUT_OF_DOMAIN'} ({result[:20]})")
        return is_in_domain
    except Exception as e:
        print(f"Query classification failed: {e}. Defaulting to True.")
        return True