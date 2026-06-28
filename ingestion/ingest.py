import os
import shutil
import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from ingestion.parser import parse_pdf

# Load environment variables
load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
BM25_RESEARCH_PATH = Path("bm25_research.pkl")
BM25_EARNINGS_PATH = Path("bm25_earnings.pkl")

def get_namespace(pdf_path: Path) -> str:
    """Determine the namespace for a PDF file based on its path and name."""
    name = pdf_path.name.lower()
    # Check if inside earning-reports directory or matches earnings/transcript keywords
    if (pdf_path.parent.name == "earning-reports" or 
        "earning" in name or 
        "transcript" in name or 
        "nvidiaan" in name):
        return "earning-reports"
    return "research"

def load_and_parse_all() -> list[Document]:
    """Scan both directories and parse all PDFs using the layout-aware parser."""
    parsed_documents = []
    
    # Scan for PDFs in both namespace directories
    pdf_files = list(DATA_DIR.glob("**/*.pdf"))
    print(f"[Ingest] Found {len(pdf_files)} PDF documents to ingest.")
    
    for pdf_path in pdf_files:
        namespace = get_namespace(pdf_path)
        print(f"[Ingest] Parsing: {pdf_path.name} -> Namespace: {namespace}")
        
        try:
            # Use our layout-aware parser
            docs = parse_pdf(pdf_path)
            for doc in docs:
                doc.metadata["namespace"] = namespace
                # Keep source and page from parser, add parser metadata
            parsed_documents.extend(docs)
        except Exception as e:
            print(f"[Ingest] ERROR parsing {pdf_path.name}: {e}")
            
    print(f"[Ingest] Total pages parsed: {len(parsed_documents)}")
    return parsed_documents

def build_hierarchical_chunks(documents: list[Document]) -> list[Document]:
    """Split documents into parent and child chunks."""
    print("[Ingest] Building hierarchical parent-child chunks...")
    
    # Parent splitter (larger context window)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    
    # Child splitter (dense semantic search segment)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    
    child_documents = []
    parent_count = 0
    
    # Group documents by source file to create sequence parent IDs
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 1)
        namespace = doc.metadata.get("namespace", "research")
        
        # Split page content into parent chunks
        parents = parent_splitter.split_text(doc.page_content)
        
        for parent_idx, parent_text in enumerate(parents):
            parent_id = f"{source}_page{page}_p{parent_idx}"
            parent_count += 1
            
            # Split this parent into child chunks
            children = child_splitter.split_text(parent_text)
            for child_idx, child_text in enumerate(children):
                child_doc = Document(
                    page_content=child_text,
                    metadata={
                        "parent_id": parent_id,
                        "parent_content": parent_text,  # Directly embed parent text in metadata
                        "source": source,
                        "page": page,
                        "namespace": namespace,
                        "child_idx": child_idx
                    }
                )
                child_documents.append(child_doc)
                
    print(f"[Ingest] Created {parent_count} parent chunks and {len(child_documents)} child chunks.")
    return child_documents

def build_bm25_indices(child_chunks: list[Document]):
    """Build and save separate BM25 indices for each namespace."""
    print("[Ingest] Building separate BM25 indices for namespaces...")
    from rank_bm25 import BM25Okapi
    
    # Separate chunks by namespace
    research_docs = [doc for doc in child_chunks if doc.metadata["namespace"] == "research"]
    earnings_docs = [doc for doc in child_chunks if doc.metadata["namespace"] == "earning-reports"]
    
    # Build research index
    if research_docs:
        tokenized_research = [doc.page_content.lower().split() for doc in research_docs]
        bm25_research = BM25Okapi(tokenized_research)
        with open(BM25_RESEARCH_PATH, "wb") as f:
            pickle.dump({"bm25": bm25_research, "documents": research_docs}, f)
        print(f"[Ingest] BM25 Research index built over {len(research_docs)} chunks.")
        
    # Build earnings index
    if earnings_docs:
        tokenized_earnings = [doc.page_content.lower().split() for doc in earnings_docs]
        bm25_earnings = BM25Okapi(tokenized_earnings)
        with open(BM25_EARNINGS_PATH, "wb") as f:
            pickle.dump({"bm25": bm25_earnings, "documents": earnings_docs}, f)
        print(f"[Ingest] BM25 Earnings index built over {len(earnings_docs)} chunks.")

def create_vector_store(child_chunks: list[Document]):
    """Recreate the vector store with child chunks."""
    if CHROMA_DIR.exists():
        print(f"[Ingest] Deleting existing database at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)
        
    print("[Ingest] Creating embeddings and storing in Chroma...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Batch insertion is safer for Chroma
    vectorstore = Chroma.from_documents(
        documents=child_chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    print(f"[Ingest] Chroma Vector store created at {CHROMA_DIR} containing {len(child_chunks)} child chunks.")
    return vectorstore

if __name__ == "__main__":
    print("=== Starting ingestion pipeline ===")
    parsed_docs = load_and_parse_all()
    if not parsed_docs:
        print("[Ingest] No documents parsed. Ingestion aborted.")
        exit(1)
        
    child_chunks = build_hierarchical_chunks(parsed_docs)
    build_bm25_indices(child_chunks)
    create_vector_store(child_chunks)
    print("=== Ingestion complete ===")
