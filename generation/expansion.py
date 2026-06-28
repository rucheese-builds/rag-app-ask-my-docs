from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

EXPANSION_PROMPT = """You are a domain-specific query expansion assistant. Your task is to expand the user's search query to improve vector database retrieval. 

CRITICAL: You must ONLY expand terms using synonyms, technical concepts, and acronym explanations that directly relate to:
1. AI agent architectures and multi-agent system frameworks (e.g. CAMEL, AutoGen, AgentVerse, ReAct).
2. Enterprise AI strategies and business adoption transcripts (Salesforce Agentforce, Microsoft, ServiceNow, Nvidia, Google, IBM earnings calls).

Do not write a long paragraph. Return a single sentence representing the expanded query. Do not explain words outside of these domains.

Example:
Input: "What is CAMEL?"
Output: "What is CAMEL multi-agent communicative framework role-playing platform?"

Input: "How is Salesforce monetizing AI?"
Output: "How is Salesforce monetizing Agentforce and AI agents through seats, consumption-based flex credits, and enterprise app deployments?"

Input: "{query}"
Output:"""

class QueryExpansionNode:
    def __init__(self, model_name="mistral"):
        print(f"[Expansion] Loading Query Expansion LLM: {model_name}")
        self.llm = OllamaLLM(model=model_name)
        self.prompt = PromptTemplate.from_template(EXPANSION_PROMPT)
        self.chain = self.prompt | self.llm

    def expand(self, query: str) -> str:
        query_stripped = query.strip()
        if not query_stripped:
            return ""
            
        print(f"[Expansion] Expanding query: '{query_stripped}'")
        try:
            expanded = self.chain.invoke({"query": query_stripped}).strip()
            # Clean up potential LLM quotes or prefixes
            expanded = expanded.replace('"', '').replace("'", "")
            print(f"[Expansion] Expanded result: '{expanded}'")
            return expanded
        except Exception as e:
            print(f"[Expansion] Failed to expand query: {e}. Returning original.")
            return query_stripped

if __name__ == "__main__":
    node = QueryExpansionNode()
    
    test_queries = [
        "What is CAMEL?",
        "How is Salesforce monetizing AI?",
        "What is test-time compute?"
    ]
    
    for q in test_queries:
        print("\n" + "="*40)
        node.expand(q)
