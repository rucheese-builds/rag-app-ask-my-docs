import streamlit as st
import sys
import pickle
import csv
import re
from pathlib import Path
from langchain_ollama import OllamaLLM

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(
    page_title="Golden Dataset Vetting Tool",
    page_icon="✍️",
    layout="wide"
)

# Styling identical to AgentPulse for a cohesive feel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: #0B0F19 !important;
    color: #F1F5F9 !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1300px;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #FFFFFF !important;
}
.header-container {
    padding-bottom: 1rem;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 2rem;
}
.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.glass-card {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin-bottom: 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

CSV_PATH = Path("evaluation/golden_dataset.csv")

# Helper to load all parent chunks
def load_parent_chunks():
    docs = []
    # Load research documents
    research_path = Path("bm25_research.pkl")
    if research_path.exists():
        with open(research_path, "rb") as f:
            data = pickle.load(f)
            docs.extend(data["documents"])
    
    # Load earnings documents
    earnings_path = Path("bm25_earnings.pkl")
    if earnings_path.exists():
        with open(earnings_path, "rb") as f:
            data = pickle.load(f)
            docs.extend(data["documents"])
            
    # Deduplicate by parent_id
    parents = {}
    for doc in docs:
        parent_id = doc.metadata.get("parent_id")
        if parent_id and parent_id not in parents:
            parents[parent_id] = {
                "parent_id": parent_id,
                "parent_content": doc.metadata.get("parent_content"),
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page", 1),
                "namespace": doc.metadata.get("namespace", "research")
            }
    return list(parents.values())

# Helper to call LLM for draft generation
def generate_draft_qa(category, context):
    llm = OllamaLLM(model="mistral", temperature=0.7)
    
    if category == "Normal":
        prompt = f"""You are an expert Q&A generator. Given the following document section, write a single clear, factual search query/question and its correct ground truth answer.
Also, extract 3 to 5 key mandatory keywords (comma-separated) representing the essential facts (dates, numbers, entities) that MUST be present in any correct answer.

Document Section:
{context}

Format your output EXACTLY as follows:
QUESTION: [factual question]
ANSWER: [concise answer in 2-3 sentences]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
"""
    elif category == "Complex & Distractor":
        prompt = f"""You are an expert Q&A generator. Given the following document section, generate a question that looks related but represents a semantic trap, an ambiguous phrasing, or queries a distractor term present in the text.
Write the correct answer indicating the clarification or the exact factual resolution.
Also, extract 3 to 5 key mandatory keywords (comma-separated) representing the essential words that MUST be present in a correct answer.

Document Section:
{context}

Format your output EXACTLY as follows:
QUESTION: [ambiguous or distractor question]
ANSWER: [answer resolving the ambiguity]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
"""
    elif category == "Multi-hop":
        prompt = f"""You are an expert Q&A generator. Given the following document section, write a multi-hop question that requires synthesizing multiple distinct facts or numbers mentioned in the section to answer.
Write a comprehensive, correct answer.
Also, extract 3 to 5 key mandatory keywords (comma-separated) representing the essential facts that MUST be present in a correct answer.

Document Section:
{context}

Format your output EXACTLY as follows:
QUESTION: [multi-hop synthesis question]
ANSWER: [synthesis answer in 2-3 sentences]
KEYWORDS: [keyword1, keyword2, keyword3, ...]
"""
    else: # Negative
        prompt = f"""You are an expert Q&A generator. Given the following document section, write a query that is semantically related to the topic of this section but is NOT covered or cannot be answered using this document.
The answer MUST be: "This question is outside the scope of the document corpus."
The keywords MUST be: "outside the scope, corpus"

Document Section:
{context}

Format your output EXACTLY as follows:
QUESTION: [out-of-corpus question that sounds plausible]
ANSWER: This question is outside the scope of the document corpus.
KEYWORDS: outside the scope, corpus
"""

    try:
        response = llm.invoke(prompt)
        # Parse fields
        q_match = re.search(r"QUESTION:\s*(.*?)(?=\nANSWER:|$)", response, re.DOTALL)
        a_match = re.search(r"ANSWER:\s*(.*?)(?=\nKEYWORDS:|$)", response, re.DOTALL)
        k_match = re.search(r"KEYWORDS:\s*(.*?)$", response, re.DOTALL)
        
        q = q_match.group(1).strip() if q_match else ""
        a = a_match.group(1).strip() if a_match else ""
        k = k_match.group(1).strip() if k_match else ""
        
        # Clean potential brackets
        q = re.sub(r"^\[|\]$", "", q)
        a = re.sub(r"^\[|\]$", "", a)
        k = re.sub(r"^\[|\]$", "", k)
        
        return q, a, k
    except Exception as e:
        st.error(f"Error calling local Mistral: {e}")
        return "", "", ""

# Save vetted Q&A to CSV
def save_vetted_qa(question, answer, keywords, category, source, page, parent_id):
    fieldnames = ["question", "ground_truth", "mandatory_keywords", "category", "relevant_sources", "parent_chunk_id", "page_number"]
    file_exists = CSV_PATH.exists()
    
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "question": question.strip(),
            "ground_truth": answer.strip(),
            "mandatory_keywords": keywords.strip(),
            "category": category,
            "relevant_sources": source,
            "parent_chunk_id": parent_id,
            "page_number": str(page)
        })

# Load CSV dataset
def load_dataset():
    if not CSV_PATH.exists():
        return []
    records = []
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records

# Delete record from CSV
def delete_record(index_to_delete):
    records = load_dataset()
    if 0 <= index_to_delete < len(records):
        records.pop(index_to_delete)
        fieldnames = ["question", "ground_truth", "mandatory_keywords", "category", "relevant_sources", "parent_chunk_id", "page_number"]
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        return True
    return False

# Main Streamlit App UI
st.markdown('<div class="header-container"><h1 class="header-title">✍️ Golden Q&A Dataset Generator</h1></div>', unsafe_allow_html=True)

# Initialize Session State
if "draft_question" not in st.session_state:
    st.session_state.draft_question = ""
if "draft_answer" not in st.session_state:
    st.session_state.draft_answer = ""
if "draft_keywords" not in st.session_state:
    st.session_state.draft_keywords = ""

parent_chunks = load_parent_chunks()

if not parent_chunks:
    st.warning("No parent chunks found. Please build the indexes first by running `python ingestion/ingest.py`.")
else:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card"><h3>1. Choose Source Context</h3></div>', unsafe_allow_html=True)
        
        # Filters
        namespaces = sorted(list(set(c["namespace"] for c in parent_chunks)))
        sel_namespace = st.selectbox("Namespace", ["All"] + namespaces)
        
        filtered_chunks = parent_chunks
        if sel_namespace != "All":
            filtered_chunks = [c for c in filtered_chunks if c["namespace"] == sel_namespace]
            
        sources = sorted(list(set(c["source"] for c in filtered_chunks)))
        sel_source = st.selectbox("Document", sources)
        
        filtered_chunks = [c for c in filtered_chunks if c["source"] == sel_source]
        
        # Chunk Selector
        chunk_options = [f"Page {c['page']} (ID: {c['parent_id']})" for c in filtered_chunks]
        selected_idx = st.selectbox("Chunk Selector", range(len(chunk_options)), format_func=lambda x: chunk_options[x])
        
        selected_chunk = filtered_chunks[selected_idx]
        
        # Display Context
        st.text_area("Parent Chunk Text", selected_chunk["parent_content"], height=300, disabled=True)
        st.caption(f"Namespace: `{selected_chunk['namespace']}` | Source: `{selected_chunk['source']}` | Page: `{selected_chunk['page']}`")
        
    with col2:
        st.markdown('<div class="glass-card"><h3>2. Generate & Vet Q&A Pair</h3></div>', unsafe_allow_html=True)
        
        category = st.selectbox("Question Category", ["Normal", "Complex & Distractor", "Multi-hop", "Negative"])
        
        if st.button("🪄 Draft Q&A with local Mistral"):
            with st.spinner("Drafting question and answer using local model..."):
                q, a, k = generate_draft_qa(category, selected_chunk["parent_content"])
                st.session_state.draft_question = q
                st.session_state.draft_answer = a
                st.session_state.draft_keywords = k
                st.toast("Draft generated! Tweak the text fields below if needed.", icon="✨")
                
        # Vetting Form
        vetted_q = st.text_input("Vetted Question", value=st.session_state.draft_question)
        vetted_a = st.text_area("Vetted Ground Truth Answer", value=st.session_state.draft_answer, height=120)
        vetted_k = st.text_input("Mandatory Keywords (comma-separated)", value=st.session_state.draft_keywords)
        
        st.caption("🚨 **Note**: Keywords are case-insensitive and used to match facts. Make sure they are distinct and crucial to the answer.")
        
        if st.button("✅ Accept & Save to Dataset", type="primary"):
            if not vetted_q or not vetted_a or not vetted_k:
                st.error("Question, Ground Truth Answer, and Keywords cannot be empty!")
            else:
                save_vetted_qa(
                    question=vetted_q,
                    answer=vetted_a,
                    keywords=vetted_k,
                    category=category,
                    source=selected_chunk["source"],
                    page=selected_chunk["page"],
                    parent_id=selected_chunk["parent_id"]
                )
                st.success("Successfully added to golden_dataset.csv!")
                st.session_state.draft_question = ""
                st.session_state.draft_answer = ""
                st.session_state.draft_keywords = ""
                # Clear st cache to reload dataset
                st.rerun()

st.markdown("---")
st.markdown('<div class="glass-card"><h3>📚 Current Golden Dataset</h3></div>', unsafe_allow_html=True)

dataset = load_dataset()
if not dataset:
    st.info("No vetted questions in the golden dataset yet. Generate some above!")
else:
    st.write(f"Total Saved Questions: **{len(dataset)}**")
    
    # Render interactive delete interface
    for idx, row in enumerate(dataset):
        with st.expander(f"Q: {row['question']} [{row['category']}]"):
            st.markdown(f"**Ground Truth**: {row['ground_truth']}")
            st.markdown(f"**Mandatory Keywords**: `{row['mandatory_keywords']}`")
            st.markdown(f"**Source**: `{row['relevant_sources']}` | **Page**: {row['page_number']}")
            
            if st.button("🗑️ Delete Question", key=f"del_{idx}"):
                if delete_record(idx):
                    st.success("Deleted question!")
                    st.rerun()
