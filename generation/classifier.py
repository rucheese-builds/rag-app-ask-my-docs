DOMAIN_KEYWORDS = [
    "agentforce", "multi-agent", "multiagent", "web of agents",
    "autogen", "agentverse", "camel", "react agent", "toolformer",
    "langchain agent", "agent communication", "agent coordination",
    "agent orchestration", "agent network", "agent ranking",
    "agentic", "llm agent", "ai agent", "agent framework",
    "internet of agents", "agent hospital", "agent bench",
    "openagents", "salesforce agent", "microsoft copilot",
    "servicenow ai", "nvidia ai", "earnings call", "agentforce"
]

def classify_query(query, llm=None):
    query_lower = query.lower()
    matched = [kw for kw in DOMAIN_KEYWORDS if kw in query_lower]
    is_in_domain = len(matched) > 0
    print(f"Query classification: {'IN_DOMAIN' if is_in_domain else 'OUT_OF_DOMAIN'}")
    if matched:
        print(f"Matched keywords: {matched}")
    return is_in_domain