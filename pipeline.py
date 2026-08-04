import os
from typing import Annotated, List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END

from generation.classifier import classify_query
from generation.expansion import QueryExpansionNode
from retrieval.retriever import load_vector_store, load_bm25_indices, hybrid_search
from reranking.reranker import load_reranker, rerank
from generation.generator import load_llm, generate_answer

# Define state schema for LangGraph
class RAGState(TypedDict):
    query: str
    expanded_query: str
    chat_history: List[Dict[str, str]]
    retrieved_chunks: List[Any]
    answer: str
    sources: List[Dict[str, Any]]
    in_domain: bool
    mode: str
    namespace: str
    alpha: float
    use_expansion: bool
    use_reranker: bool

# Global variables for loaded assets
global_vectorstore = None
global_bm25_indices = None
global_reranker = None
global_llm = None
global_expansion_node = None
global_neo4j_query_engine = None

OUT_OF_DOMAIN_RESPONSE = """This question appears to be outside the scope
of my document corpus, which covers:

- Web of Agents research papers
- Multi-agent system architectures
- Enterprise AI adoption (Salesforce, Microsoft, Nvidia, ServiceNow, Google, IBM)

Please ask something related to these topics."""

def load_pipeline():
    global global_vectorstore, global_bm25_indices, global_reranker, global_llm, global_expansion_node
    print("[Pipeline] Loading pipeline components...")
    global_vectorstore = load_vector_store()
    global_bm25_indices = load_bm25_indices()
    global_reranker = load_reranker()
    global_llm = load_llm()
    global_expansion_node = QueryExpansionNode()
    print("[Pipeline] Pipeline ready.")
    return global_vectorstore, global_bm25_indices, global_reranker, global_llm, global_expansion_node

def load_neo4j_pipeline():
    global global_neo4j_query_engine
    import os
    from llama_index.core import PropertyGraphIndex
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.ollama import Ollama

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not neo4j_password:
        raise ValueError("NEO4J_PASSWORD is not set in your .env file!")

    print("[Pipeline] Connecting to Neo4j Property Graph (Read-Only)...")
    graph_store = Neo4jPropertyGraphStore(
        url=neo4j_uri,
        username=neo4j_username,
        password=neo4j_password,
        database=neo4j_database
    )

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = Ollama(
        model="mistral",
        base_url=ollama_base_url,
        request_timeout=120.0
    )

    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        embed_model=HuggingFaceEmbedding(
            model_name=os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5"),
            cache_folder="./hf_cache"
        ),
        llm=llm
    )
    
    print("[Pipeline] Neo4j read-only query engine loaded successfully (using local Mistral).")
    global_neo4j_query_engine = index.as_query_engine(use_async=False)
    return global_neo4j_query_engine

# --- LangGraph Nodes ---

def classifier_node(state: RAGState) -> Dict[str, Any]:
    print("[LangGraph Node] Classifier")
    query = state["query"]
    in_domain = classify_query(query, global_llm)
    return {"in_domain": in_domain}

def fallback_node(state: RAGState) -> Dict[str, Any]:
    print("[LangGraph Node] Gemini Fallback")
    query = state["query"]
    chat_history = state.get("chat_history", [])
    
    import google.generativeai as genai
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"answer": "Error: GEMINI_API_KEY is not set in environment.", "in_domain": False}
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        history_str = ""
        if chat_history:
            history_parts = []
            for msg in chat_history[-5:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_parts.append(f"{role}: {msg['content']}")
            history_str = "Conversation History:\n" + "\n".join(history_parts) + "\n\n"
            
        prompt = f"""{history_str}The user asked a question that is outside the scope of our local document database.
Please answer the query using your general knowledge:
Query: {query}"""
        
        response = model.generate_content(prompt)
        return {
            "answer": response.text.strip(),
            "in_domain": False
        }
    except Exception as e:
        return {"answer": f"Error calling Gemini: {e}", "in_domain": False}

def vector_rag_node(state: RAGState) -> Dict[str, Any]:
    print("[LangGraph Node] Vector RAG retrieval and synthesis")
    query = state["query"]
    chat_history = state.get("chat_history", [])
    namespace = state.get("namespace", "all")
    alpha = state.get("alpha", 0.4)
    use_expansion = state.get("use_expansion", True)
    use_reranker = state.get("use_reranker", True)
    
    # 1. Expand query
    expanded_query = query
    if use_expansion and global_expansion_node:
        expanded_query = global_expansion_node.expand(query)
        
    # 2. Hybrid search
    retrieved_child_chunks = hybrid_search(
        global_vectorstore,
        global_bm25_indices,
        expanded_query,
        namespace=namespace,
        alpha=alpha,
        k=20
    )
    
    if not retrieved_child_chunks:
        return {
            "answer": "I couldn't find any relevant documents to answer your question.",
            "sources": [],
            "expanded_query": expanded_query
        }
        
    # 3. Rerank
    if use_reranker and global_reranker:
        reranked_child_chunks = rerank(global_reranker, expanded_query, retrieved_child_chunks, top_n=3)
    else:
        reranked_child_chunks = retrieved_child_chunks[:3]
        
    # 4. Context swap
    generation_chunks = []
    for doc in reranked_child_chunks:
        parent_content = doc.metadata.get("parent_content")
        if parent_content:
            parent_doc = doc.__class__(
                page_content=parent_content,
                metadata=doc.metadata
            )
            generation_chunks.append(parent_doc)
        else:
            generation_chunks.append(doc)
            
    # 5. Synthesis via local Mistral with memory
    answer = generate_answer(global_llm, query, generation_chunks, chat_history=chat_history)
    
    # 6. Format sources
    sources = []
    for i, doc in enumerate(reranked_child_chunks):
        sources.append({
            "index": i + 1,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "unknown"),
            "namespace": doc.metadata.get("namespace", "unknown"),
            "content": doc.page_content,
            "parent_content": doc.metadata.get("parent_content", doc.page_content)[:1500]
        })
        
    return {
        "answer": answer,
        "sources": sources,
        "expanded_query": expanded_query
    }

def graph_rag_node(state: RAGState) -> Dict[str, Any]:
    print("[LangGraph Node] Graph RAG retrieval and synthesis")
    query = state["query"]
    chat_history = state.get("chat_history", [])
    
    condensed_query = query
    if chat_history:
        from llama_index.core.llms import ChatMessage
        history_msgs = []
        for msg in chat_history[-5:]:
            role = "user" if msg["role"] == "user" else "assistant"
            history_msgs.append(ChatMessage(role=role, content=msg["content"]))
            
        condense_prompt = f"""Given the conversation history and a follow-up query, re-phrase the follow-up query to be a standalone query.
        
History:
{" ".join([f"{m.role}: {m.content}" for m in history_msgs])}

Follow-up Query: {query}
Standalone Query:"""
        try:
            condensed_query = str(global_neo4j_query_engine._llm.complete(condense_prompt)).strip()
            print(f"[Graph RAG] Condensed query: {condensed_query}")
        except Exception as e:
            print(f"[Graph RAG] Failed to condense query: {e}")
            condensed_query = query

    response = global_neo4j_query_engine.query(condensed_query)
    
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
        "sources": sources
    }

# --- Routing Logic ---

def route_classifier(state: RAGState) -> str:
    if not state.get("in_domain", True):
        return "fallback"
        
    mode = state.get("mode", "vector")
    if mode == "graph":
        return "graph_rag"
    else:
        return "vector_rag"

# --- Graph Compilation ---

def compile_rag_graph():
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("vector_rag", vector_rag_node)
    workflow.add_node("graph_rag", graph_rag_node)
    workflow.add_node("fallback", fallback_node)
    
    # Set entry point
    workflow.set_entry_point("classifier")
    
    # Add conditional router edge
    workflow.add_conditional_edges(
        "classifier",
        route_classifier,
        {
            "vector_rag": "vector_rag",
            "graph_rag": "graph_rag",
            "fallback": "fallback"
        }
    )
    
    # Connect leaf nodes to END
    workflow.add_edge("vector_rag", END)
    workflow.add_edge("graph_rag", END)
    workflow.add_edge("fallback", END)
    
    return workflow.compile()

# Compile global state graph
rag_graph = compile_rag_graph()

# --- Compatible Entry Points ---

def run_pipeline(query, vectorstore, bm25_indices, reranker, llm, expansion_node, 
                 namespace="all", alpha=0.5, use_expansion=True, use_reranker=True, chat_history=None):
    global global_vectorstore, global_bm25_indices, global_reranker, global_llm, global_expansion_node
    global_vectorstore = vectorstore
    global_bm25_indices = bm25_indices
    global_reranker = reranker
    global_llm = llm
    global_expansion_node = expansion_node

    inputs = {
        "query": query,
        "chat_history": chat_history or [],
        "mode": "vector",
        "namespace": namespace,
        "alpha": alpha,
        "use_expansion": use_expansion,
        "use_reranker": use_reranker
    }
    
    output = rag_graph.invoke(inputs)
    
    return {
        "answer": output.get("answer", OUT_OF_DOMAIN_RESPONSE),
        "sources": output.get("sources", []),
        "in_domain": output.get("in_domain", True),
        "query": query,
        "expanded_query": output.get("expanded_query", query),
        "_reranked_docs": []
    }

def run_neo4j_pipeline(query, query_engine, chat_history=None):
    global global_neo4j_query_engine
    global_neo4j_query_engine = query_engine
    
    inputs = {
        "query": query,
        "chat_history": chat_history or [],
        "mode": "graph"
    }
    
    output = rag_graph.invoke(inputs)
    
    return {
        "answer": output.get("answer", ""),
        "sources": output.get("sources", []),
        "in_domain": output.get("in_domain", True),
        "query": query,
        "expanded_query": query,
        "_reranked_docs": []
    }