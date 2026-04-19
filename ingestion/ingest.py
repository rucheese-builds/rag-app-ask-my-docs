import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")

PAPER_DESCRIPTIONS = {
    "Internet of Agents.pdf": "CLUSTER: Ecosystem & Internet. Proposes internet-scale architecture for connecting heterogeneous AI agents into collaborative networks. Focuses on connectivity layer and agent integration protocols.",
    "Internet 3.0.pdf": "CLUSTER: Ecosystem & Internet. Introduces machine-native web ecosystem with AgentRank algorithm for agent discovery and governance. Focuses on trustworthy large-scale agent networks.",
    "AgentVerse.pdf": "CLUSTER: Ecosystem & Internet. Multi-agent framework where agents take specialized roles (recruiter, critic, worker) in decentralized collaborative ecosystems.",
    "AutoGen.pdf": "CLUSTER: Communication & Framework. Microsoft's practical framework for multi-agent conversation, tool use, and code execution. Focuses on building LLM applications through agent collaboration.",
    "CAMEL.pdf": "CLUSTER: Communication & Framework. Academic role-playing framework for autonomous agent cooperation using natural language dialogue. Focuses on agent communication dynamics and scaling.",
    "L2M2 Multi-agent Coordination.pdf": "CLUSTER: Communication & Framework. Orchestration framework where a manager agent decomposes tasks and assigns them to specialized sub-agents. Focuses on hierarchical coordination.",
    "Dynamic LLM.pdf": "CLUSTER: Communication & Framework. Dynamic agent network where team composition adapts based on performance. Argues against fixed agent teams in favour of task-specific selection.",
    "ReAct.pdf": "CLUSTER: Capabilities & Training. Foundation framework combining reasoning and acting in LLMs through thought-action-observation loops. Underpins most modern agent architectures.",
    "Scaling LLM Google Deepmind.pdf": "CLUSTER: Capabilities & Training. Google DeepMind research on test-time compute scaling. Shows smaller models with more thinking time can outperform larger models.",
    "Scaling Agent Systems.pdf": "CLUSTER: Capabilities & Training. First quantitative scaling laws for multi-agent systems. Mathematical framework predicting performance based on agent count and coordination structure.",
    "OpenAgents.pdf": "CLUSTER: Deployment & Testing. Open platform for deploying language agents for everyday use. Focuses on user-friendly interfaces for data analysis, tool use, and web browsing.",
    "AgentBench.pdf": "CLUSTER: Deployment & Testing. Benchmark for evaluating LLM agents across 8 environments. Focuses on evaluation methodology and metrics, NOT agent network architecture.",
    
}

def extract_section_header(text):
    lines = text.strip().split('\n')
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 10 and len(line) < 100 and line[0].isupper():
            return line
    return ""

def load_documents():
    documents = []
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} documents")
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        description = PAPER_DESCRIPTIONS.get(pdf_path.name, "")
        paper_title = pdf_path.stem
        for doc in docs:
            section_header = extract_section_header(doc.page_content)
            header_prefix = f"{paper_title}"
            if section_header:
                header_prefix += f" > {section_header}"
            header_prefix += "\n\n"
            doc.page_content = header_prefix + doc.page_content
            doc.metadata["source"] = pdf_path.name
            doc.metadata["paper_description"] = description
            doc.metadata["paper_title"] = paper_title
        documents.extend(docs)
    print(f"Total pages loaded: {len(documents)}")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=128,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    print("Creating embeddings and storing in Chroma...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")
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

