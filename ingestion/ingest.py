import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

def load_documents():
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} documents")
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        documents.extend(docs)
    print(f"Total pages loaded: {len(documents)}")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    print("Creating embeddings and storing in Chroma...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    print(f"Vector store created at {CHROMA_DIR}")
    return vectorstore

if __name__ == "__main__":
    print("=== Starting ingestion pipeline ===")
    documents = load_documents()
    chunks = chunk_documents(documents)
    create_vector_store(chunks)
    print("=== Ingestion complete ===")