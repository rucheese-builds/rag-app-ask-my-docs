import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal

# Add project root to sys.path to resolve ingestion/parser.py imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file containing credentials
load_dotenv()

try:
    # 1. Import LlamaIndex Core and integrations
    from llama_index.core import Settings, PropertyGraphIndex, Document as LlamaIndexDocument
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    from llama_index.core.indices.property_graph import (
        SchemaLLMPathExtractor,
        SimpleLLMPathExtractor,
        ImplicitPathExtractor,
    )
except ImportError:
    print("\n[Error] Required packages not found. Please install LlamaIndex dependencies:")
    print("pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-graph-stores-neo4j\n")
    sys.exit(1)

def run_pipeline():
    print("=== Starting LlamaIndex Knowledge Graph Ingestion Pipeline (Local) ===")

    # 2. Config global Settings to map local Ollama and HuggingFace models
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    print(f"[Config] Initializing Local LLM: {OLLAMA_MODEL} via Ollama ({OLLAMA_BASE_URL})...")
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=360.0, # Long timeout for intensive entity extraction tasks
    )

    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    print(f"[Config] Initializing Local Embeddings: {EMBED_MODEL_NAME} via HuggingFace...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder="./hf_cache" # Stores model cache locally in project
    )

    # 3. Configure connection to local/cloud Neo4j instance
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

    if not NEO4J_PASSWORD:
        print("[Error] Missing NEO4J_PASSWORD environment variable in .env!")
        sys.exit(1)

    print(f"[Config] Connecting to Neo4j database '{NEO4J_DATABASE}' at {NEO4J_URI}...")
    graph_store = Neo4jPropertyGraphStore(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )

    # 4. Integrate Directory Loading & Pruning Logic
    # Reuses parser.py to run LlamaParse and prune bibliography/references pages.
    from ingestion.parser import parse_pdf

    RESEARCH_DIR = Path("data/research")
    EARNINGS_DIR = Path("data/earning-reports")
    
    pdf_files = []
    if RESEARCH_DIR.exists():
        pdf_files.extend(list(RESEARCH_DIR.glob("*.pdf")))
    if EARNINGS_DIR.exists():
        pdf_files.extend(list(EARNINGS_DIR.glob("*.pdf")))

    if not pdf_files:
        print(f"[Error] No PDF files found in {RESEARCH_DIR} or {EARNINGS_DIR}!")
        sys.exit(1)

    print(f"[Ingest] Found {len(pdf_files)} PDF documents to load.")
    
    documents = []
    for p in pdf_files:
        print(f"[Ingest] Parsing & Pruning: {p.name}")
        lc_docs = parse_pdf(p)
        for lc_doc in lc_docs:
            # Convert LangChain Document to LlamaIndex native Document
            llama_doc = LlamaIndexDocument(
                text=lc_doc.page_content,
                metadata={
                    "source": lc_doc.metadata.get("source", p.name),
                    "page": lc_doc.metadata.get("page", 1),
                    "namespace": lc_doc.metadata.get("namespace", "all")
                }
            )
            documents.append(llama_doc)
            
    print(f"[Ingest] Successfully loaded and pruned to {len(documents)} total document pages.")

    # 5. Define Knowledge Graph Ontological Schema
    # This prevents local LLM hallucination and maps elements to standard categories.
    Entities = Literal[
        "Framework",      # e.g., CAMEL, AgentVerse, AutoGen, AgentRank, L2M2
        "Company",        # e.g., Salesforce, Microsoft, ServiceNow, Nvidia, Google
        "Role",           # e.g., Orchestrator, Recruiter, Critic, Caller, Callee
        "Metric",         # e.g., Hit Rate, Context Precision, Fact Recall, Semantic Similarity
        "Technology",     # e.g., MCP, Vector DB, LLM, Cross-Encoder
        "Concept"         # e.g., Multi-agent collaboration, semantic trap
    ]

    Relations = Literal[
        "MENTIONS",
        "DISCUSSES",
        "COMPARES",
        "MONETIZES",
        "DELIVERS",
        "IMPLEMENTS"
    ]

    # Setup the schema-guided path extractor
    schema_extractor = SchemaLLMPathExtractor(
        llm=Settings.llm,
        possible_entities=Entities,
        possible_relations=Relations,
        strict=False # False allows fallback parsing when 8B model uses casing/syntax variations
    )

    # Combining schema extractor with syntax-based implicit extractor
    extractors = [schema_extractor, ImplicitPathExtractor()]

    # 6. Run the Ingestion Pipeline to extract entities and relations into Neo4j
    print("[Pipeline] Ingesting documents and building Property Graph...")
    print("[Pipeline] Extraction will stream directly to Neo4j. This may take some time depending on your CPU/GPU.")
    
    try:
        index = PropertyGraphIndex.from_documents(
            documents,
            property_graph_store=graph_store,
            kg_extractors=extractors,
            show_progress=True
        )
        print("\n=== Ingestion Complete! ===")
        print("You can now open Neo4j Bloom or Browser to visualize your extracted knowledge graph.")
        
        # Test Query Engine
        print("\n[Pipeline] Running a test hybrid query over Neo4j...")
        query_engine = index.as_query_engine()
        response = query_engine.query("How does Salesforce monetize Agentforce compared to academic collaboration frameworks?")
        print(f"\nResponse:\n{response}")
        
    except Exception as e:
        print(f"\n[Pipeline] Ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()
