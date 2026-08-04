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
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
    from llama_index.core.indices.property_graph import (
        SimpleLLMPathExtractor,
        ImplicitPathExtractor,
    )
except ImportError:
    print("\n[Error] Required packages not found. Please install LlamaIndex dependencies:")
    print("pip install llama-index-core llama-index-llms-openai llama-index-embeddings-huggingface llama-index-graph-stores-neo4j\n")
    sys.exit(1)

def run_pipeline():
    print("=== Starting LlamaIndex Knowledge Graph Ingestion Pipeline (Hybrid Local/Cloud) ===")

    # 2. Config global Settings: cloud OpenAI for entity extraction, local HuggingFace for embeddings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        print("[Error] Missing OPENAI_API_KEY environment variable in .env!")
        print("Please add 'OPENAI_API_KEY = \"your-api-key\"' to your .env file to enable fast cloud extraction.")
        sys.exit(1)

    print("[Config] Initializing Cloud LLM: gpt-4o-mini via OpenAI...")
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        request_timeout=120.0
    )

    EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    print(f"[Config] Initializing Local Embeddings: {EMBED_MODEL_NAME} via HuggingFace...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder="./hf_cache" # Stores model cache locally in project
    )

    # 3. Configure connection to Neo4j instance
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

    # 5. Define Knowledge Graph Prompt Instructions
    # This guides the model to use our specific schema types (Framework, Company, Role, Metric, Technology)
    # and relationships (MENTIONS, DISCUSSES, COMPARES, MONETIZES, DELIVERS, IMPLEMENTS)
    # while outputting in standard (subject, predicate, object) format for native parsing.
    custom_prompt = (
        "Some text is provided below. Given the text, extract up to {max_paths_per_chunk} "
        "knowledge triples in the form of (subject, predicate, object) representing relationships "
        "between Frameworks, Companies, Roles, Metrics, and Technologies.\n"
        "Focus on relationships like MENTIONS, DISCUSSES, COMPARES, MONETIZES, DELIVERS, and IMPLEMENTS.\n"
        "Format each triple exactly as: (subject, predicate, object) on a new line.\n"
        "Avoid pronouns and generic subjects.\n"
        "Text:\n{text}\n"
        "Triples:\n"
    )

    # Setup the robust simple path extractor (fully compatible with OpenAI gpt-4o-mini)
    simple_extractor = SimpleLLMPathExtractor(
        llm=Settings.llm,
        extract_prompt=custom_prompt,
        max_paths_per_chunk=15
    )

    # Combining simple text-based extractor with syntax-based implicit extractor
    extractors = [simple_extractor, ImplicitPathExtractor()]

    # 6. Run the Ingestion Pipeline to extract entities and relations into Neo4j
    print("[Pipeline] Ingesting documents and building Property Graph...")
    print("[Pipeline] Extraction will run via gpt-4o-mini. This should only take a few minutes.")
    
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
