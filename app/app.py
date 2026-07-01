import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import load_pipeline, run_pipeline, load_neo4j_pipeline, run_neo4j_pipeline

st.set_page_config(
    page_title="AgentLens — Local RAG Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern CSS Injection
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

/* Reset and Core Styles */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
    background: #0B0F19 !important; /* Deep space dark background */
    color: #F1F5F9 !important;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1200px;
}

/* Typography overrides */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #0F172A !important; /* Slate 900 background */
    border-right: 1px solid #1E293B !important;
}
[data-testid="stSidebar"] > div {
    padding: 2rem 1.5rem;
}
[data-testid="stSidebar"] * {
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] label {
    font-size: 0.85rem !important;
    color: #94A3B8 !important;
    font-weight: 500 !important;
}

/* Glassmorphic Cards */
.glass-card {
    background: rgba(15, 23, 42, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin-bottom: 1.2rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

.glass-card h3 {
    margin-bottom: 0.5rem;
    font-size: 1.1rem;
    background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Gradient Header */
.header-container {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1E293B;
    margin-bottom: 2rem;
}
.header-logo {
    font-size: 1.8rem;
    background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
    padding: 8px 14px;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
}
.header-title {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-subtitle {
    font-size: 0.9rem;
    color: #64748B;
    margin-top: 0.2rem;
}

/* Custom Streamlit Tabs styling */
[data-testid="stTabs"] {
    background: transparent !important;
}
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #1E293B;
    gap: 8px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important;
    background: transparent !important;
    border: none !important;
    padding: 8px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #94A3B8 !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #1E293B !important;
    color: #6366F1 !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
}

/* Inactive suggested Q buttons */
.suggested-q-btn > div > button {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: #CBD5E1 !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    text-align: left !important;
    font-size: 0.85rem !important;
}
.suggested-q-btn > div > button:hover {
    background: rgba(30, 41, 59, 0.8) !important;
    border-color: #6366F1 !important;
    color: #FFFFFF !important;
}

/* Beautiful Answer Box */
.answer-box {
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%) !important;
    border: 1.5px solid rgba(99, 102, 241, 0.3) !important;
    border-left: 5px solid #6366F1 !important;
    border-radius: 12px;
    padding: 1.5rem;
    line-height: 1.8;
    color: #F1F5F9;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Source Badges */
.badge-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.5rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid #334155;
    border-radius: 30px;
    padding: 0.3rem 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #94A3B8;
}
.badge.research {
    border-color: rgba(99, 102, 241, 0.5);
    color: #818CF8;
    background: rgba(99, 102, 241, 0.1);
}
.badge.earnings {
    border-color: rgba(16, 185, 129, 0.5);
    color: #34D399;
    background: rgba(16, 185, 129, 0.1);
}

/* Visualizer Flowchart */
.vis-container {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
}
.vis-row {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.vis-node {
    flex: 1;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 0.6rem;
    text-align: center;
    font-size: 0.8rem;
}
.vis-node.highlight {
    background: rgba(99, 102, 241, 0.2);
    border-color: #6366F1;
    font-weight: 600;
}
.vis-arrow {
    color: #64748B;
    font-size: 1.2rem;
    display: flex;
    justify-content: center;
    align-items: center;
}

/* Metric Display Card */
.metric-box {
    position: relative;
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.metric-box.has-tooltip {
    cursor: help;
}
.metric-box .tooltip-text {
    visibility: hidden;
    width: 220px;
    background: rgba(15, 23, 42, 0.95) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    color: #E2E8F0 !important;
    text-align: center;
    border-radius: 8px;
    padding: 0.8rem !important;
    position: absolute;
    z-index: 99999 !important;
    bottom: 115%;
    left: 50%;
    transform: translateX(-50%) translateY(10px);
    opacity: 0;
    transition: opacity 0.2s ease, transform 0.2s ease, visibility 0.2s ease;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    font-size: 0.75rem !important;
    line-height: 1.4 !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.6) !important;
    pointer-events: none;
    font-family: 'Inter', sans-serif !important;
    font-weight: 400 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
.metric-box:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
    transform: translateX(-50%) translateY(0);
}
.metric-box .tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -6px;
    border-width: 6px;
    border-style: solid;
    border-color: rgba(15, 23, 42, 0.95) transparent transparent transparent;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #6366F1;
}
.metric-lbl {
    font-size: 0.75rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Domain suggestions dataset (updated with namespaces and paper links)
DOMAINS = {
    "🌐 System Architecture": {
        "namespace": "research",
        "description": "Distributed agent protocols",
        "questions": [
            "How does the Internet of Agents differ from traditional multi-agent systems?",
            "What is the AgentRank algorithm in Internet 3.0?",
            "How does AgentVerse structure collaborative agent ecosystems?",
        ],
        "papers": [
            {"name": "Internet of Agents", "url": "https://arxiv.org/abs/2407.07061"},
            {"name": "AgentVerse", "url": "https://arxiv.org/abs/2308.10848"},
            {"name": "Internet 3.0", "url": "https://arxiv.org/abs/2509.04979"},
            {"name": "OpenAgents", "url": "https://arxiv.org/abs/2310.10634"}
        ]
    },
    "📡 Agent Collaboration": {
        "namespace": "research",
        "description": "Conversation and orchestration",
        "questions": [
            "How does AutoGen handle multi-agent conversation?",
            "What is role-playing in the CAMEL framework?",
            "How does L2M2 orchestrate sub-agents for complex tasks?",
        ],
        "papers": [
            {"name": "Autogen", "url": "https://arxiv.org/abs/2308.08155"},
            {"name": "CAMEL", "url": "https://arxiv.org/abs/2303.17760"},
            {"name": "L2M2 multi-agent coordination", "url": "https://arxiv.org/abs/2502.14743"}
        ]
    },
    "🧠 Reasoning & Scaling": {
        "namespace": "research",
        "description": "Reasoning loops and scaling laws",
        "questions": [
            "How does ReAct combine reasoning and acting in language models?",
            "What are the scaling laws for multi-agent systems?",
            "How does test-time compute scaling improve agent performance?",
        ],
        "papers": [
            {"name": "ReACT", "url": "https://arxiv.org/abs/2210.03629"},
            {"name": "Scaling Agent systems Google Deepmind", "url": "https://arxiv.org/abs/2512.08296v2"},
            {"name": "Scaling LLM Google Deepmind", "url": "https://arxiv.org/abs/2408.03314"},
            {"name": "Dynamic LLM", "url": "https://arxiv.org/abs/2310.02170"},
            {"name": "AgentBench", "url": "https://arxiv.org/abs/2308.03688"}
        ]
    },
    "💰 Enterprise AI Adoption": {
        "namespace": "earning-reports",
        "description": "Earnings transcripts analysis",
        "questions": [
            "How is Salesforce monetizing Agentforce?",
            "What did Microsoft say about enterprise agentic AI tools?",
            "How many deals has Salesforce Agentforce closed in its first 15 months?",
        ],
        "papers": [
            {"name": "Salesforce IR", "url": "https://investor.salesforce.com"},
            {"name": "Microsoft IR", "url": "https://www.microsoft.com/investor"},
            {"name": "NVIDIA IR", "url": "https://investor.nvidia.com"},
            {"name": "ServiceNow IR", "url": "https://investor.servicenow.com"},
            {"name": "Google IR", "url": "https://abc.xyz/investor"},
            {"name": "IBM IR", "url": "https://www.ibm.com/investor"}
        ]
    },
}

# Load the upgraded RAG pipeline
@st.cache_resource
def get_pipeline():
    return load_pipeline()

vectorstore, bm25_indices, reranker, llm, expansion_node = get_pipeline()

@st.cache_resource
def get_neo4j_pipeline():
    import os
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("NEO4J_PASSWORD"):
        return None
    try:
        return load_neo4j_pipeline()
    except Exception as e:
        print(f"Error loading Neo4j pipeline: {e}")
        return None

neo4j_query_engine = get_neo4j_pipeline()

# ── Sidebar Configurations ──
with st.sidebar:
    st.markdown("## ⚡ AgentLens")
    st.markdown("<p style='color: #64748B; font-size: 0.8rem; margin-top:-0.5rem;'>Advanced RAG Intelligence Platform</p>", unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### Search Settings")
    
    # RAG Mode selector
    rag_mode = st.selectbox(
        "RAG Engine Mode",
        options=["Hybrid Vector RAG (Chroma + BM25)", "Knowledge Graph RAG (Neo4j)"]
    )
    
    if rag_mode == "Knowledge Graph RAG (Neo4j)" and neo4j_query_engine is None:
        st.warning("⚠️ Neo4j or OpenAI credentials missing/invalid in .env! Falling back to Hybrid RAG.")
        rag_mode = "Hybrid Vector RAG (Chroma + BM25)"
        
    # 1. Namespace Selector
    namespace_mode = st.selectbox(
        "Corpus Namespace",
        options=["all", "research", "earning-reports"],
        format_func=lambda x: {
            "all": "All Documents",
            "research": "Research Papers (/research)",
            "earning-reports": "Earnings Calls (/earning-reports)"
        }[x]
    )
    
    # 2. Query Expansion Toggle
    use_expansion = st.toggle("Enable Query Expansion", value=True, help="Run queries through Mistral to expand contextual keywords before database lookup.")
    
    # 3. Alpha Slider (Convex Fusion Weight)
    alpha = st.slider(
        "Convex Weight (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="α = 1.0 (pure Vector Search) | α = 0.0 (pure Keyword BM25). Set lower (e.g. 0.4) to prioritize precise keywords and technical acronyms."
    )
    
    st.divider()
    
    # Database Stats
    st.markdown("### Database Indices")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"<div class='metric-box'><div class='metric-val'>12</div><div class='metric-lbl'>Research</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-box'><div class='metric-val'>13</div><div class='metric-lbl'>Earnings</div></div>", unsafe_allow_html=True)
        
    st.markdown(f"<div class='metric-box' style='margin-top:0.5rem'><div class='metric-val'>{vectorstore._collection.count()}</div><div class='metric-lbl'>Parent-Child Indexed Nodes</div></div>", unsafe_allow_html=True)
    
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #64748B; font-size: 0.8rem;'>
        <a href="https://github.com/rucheese-builds/rag-app-ask-my-docs" target="_blank" style="color: #CBD5E1; text-decoration: none; display: inline-flex; align-items: center; gap: 5px; font-weight: 500;">
            <svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor" style="vertical-align: middle;">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
            </svg>
            GitHub Repository
        </a>
        <div style='margin-top: 5px;'>⚡ Built By: <b>Ruchi Agarwal</b></div>
    </div>
    """, unsafe_allow_html=True)

# ── Header ──
st.markdown("""
<div class='header-container'>
    <div class='header-logo'>⚡</div>
    <div>
        <div class='header-title'>AgentLens</div>
        <div class='header-subtitle' style='margin-bottom: 8px;'>A local RAG system designed to query a corpus of the latest academic research on Web-of-Agents alongside earnings calls from top enterprise AI companies</div>
        <div style="display: flex; align-items: center; gap: 10px; font-size: 0.8rem; color: #64748B;">
            <a href="https://github.com/rucheese-builds/rag-app-ask-my-docs" target="_blank" style="display: flex; align-items: center; gap: 5px; color: #94A3B8; text-decoration: none; font-weight: 500;">
                <svg height="14" width="14" viewBox="0 0 16 16" fill="currentColor" style="vertical-align: middle;">
                    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                </svg>
                GitHub Repository
            </a>
            <span>•</span>
            <span>⚡ Built By: <b>Ruchi Agarwal</b></span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load dynamic suggestions from golden dataset
import csv
import pandas as pd

def load_golden_suggestions():
    csv_path = Path("evaluation/golden_dataset.csv")
    if not csv_path.exists():
        return []
    records = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except Exception as e:
        print(f"Error loading golden suggestions: {e}")
    return records

def load_eval_results():
    results_path = Path("evaluation/eval_results_golden.csv")
    if not results_path.exists():
        return None
    try:
        df = pd.read_csv(results_path)
        return df
    except Exception as e:
        print(f"Error loading eval results: {e}")
        return None

# ── Tabs ──
explorer_tab, benchmark_tab, about_tab = st.tabs([
    "🔍 Contextual Explorer", 
    "📊 Benchmarking Dashboard", 
    "ℹ️ Engine Architecture"
])

# Initialize session state variables
if "selected_query" not in st.session_state:
    st.session_state.selected_query = ""
if "run_trigger" not in st.session_state:
    st.session_state.run_trigger = False

# Explorer Tab
with explorer_tab:
    st.markdown("### 🔍 Explore Suggested Topics")
    col1, col2 = st.columns([1, 1])
    
    def render_domain_expander(domain_name, domain_info):
        with st.expander(f"{domain_name} — {domain_info['description']}"):
            for q in domain_info["questions"]:
                st.markdown("<div class='suggested-q-btn'>", unsafe_allow_html=True)
                if st.button(q, key=f"q_preset_{domain_name[:5]}_{q[:30]}"):
                    st.session_state.selected_query = q
                    st.session_state.run_trigger = True
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
                
            if "papers" in domain_info:
                st.markdown("<hr style='margin: 10px 0 5px 0; opacity: 0.15;'/>", unsafe_allow_html=True)
                st.markdown("<p style='color: #64748B; font-size: 0.72rem; margin-bottom: 5px; font-weight: 600; letter-spacing: 0.03em;'>📄 COVERED SOURCES:</p>", unsafe_allow_html=True)
                links = []
                for p in domain_info["papers"]:
                    links.append(f"<a href='{p['url']}' target='_blank' style='color: #6366F1; text-decoration: none; font-size: 0.75rem; font-weight: 500;'>🔗 {p['name']}</a>")
                st.markdown(" &nbsp;•&nbsp; ".join(links), unsafe_allow_html=True)

    presets = list(DOMAINS.items())
    
    with col1:
        # First 2 presets
        for domain_name, domain_info in presets[:2]:
            render_domain_expander(domain_name, domain_info)
                    
        # 3rd preset
        domain_name, domain_info = presets[2]
        render_domain_expander(domain_name, domain_info)
                
    with col2:
        # 4th preset
        domain_name, domain_info = presets[3]
        render_domain_expander(domain_name, domain_info)
                
        # 5th expander: dynamic vetted Q&As as a sub-topic
        with st.expander("✨ Verified Examples — AI-generated & human-vetted"):
            golden_qs = load_golden_suggestions()
            if not golden_qs:
                st.info("No vetted examples available yet. Use the Vetting Tool (port 8502) to add some!")
            else:
                for idx, item in enumerate(golden_qs):
                    q = item["question"]
                    st.markdown("<div class='suggested-q-btn'>", unsafe_allow_html=True)
                    if st.button(q, key=f"q_vetted_{idx}_{q[:20]}"):
                        st.session_state.selected_query = q
                        st.session_state.run_trigger = True
                        st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    
    # 2. Query Search bar
    st.markdown("### Ask your own question")
    typed_query = st.text_input(
        "Enter query",
        value=st.session_state.selected_query,
        placeholder="e.g. What makes CAMEL different from other frameworks?",
        label_visibility="collapsed"
    )
    
    col_search, col_clear = st.columns([6, 1])
    with col_search:
        search_clicked = st.button("⚡ Execute Pipeline Query")
    with col_clear:
        if st.button("🗑️ Clear", type="secondary"):
            st.session_state.selected_query = ""
            st.session_state.run_trigger = False
            st.rerun()

    # 3. Running the pipeline
    active_query = typed_query.strip()
    if (search_clicked or st.session_state.run_trigger) and active_query:
        st.session_state.run_trigger = False # reset
        
        if rag_mode == "Knowledge Graph RAG (Neo4j)":
            with st.spinner("Retrieving facts and traversing Property Graph in Neo4j..."):
                result = run_neo4j_pipeline(active_query, neo4j_query_engine)
        else:
            with st.spinner("Processing through 5-stage pipeline..."):
                result = run_pipeline(
                    active_query,
                    vectorstore,
                    bm25_indices,
                    reranker,
                    llm,
                    expansion_node,
                    namespace=namespace_mode,
                    alpha=alpha,
                    use_expansion=use_expansion,
                    use_reranker=True
                )
            
        # Display results
        if not result["in_domain"]:
            st.error(result["answer"])
        else:
            st.markdown("### Answer")
            st.markdown(f"<div class='answer-box'>{result['answer']}</div>", unsafe_allow_html=True)
            
            # Display source badges
            if result["sources"]:
                st.markdown("### Cited Sources")
                st.markdown("<div class='badge-container'>", unsafe_allow_html=True)
                badges_html = ""
                for s in result["sources"]:
                    ns_class = "earnings" if s["namespace"] == "earning-reports" else "research"
                    ns_icon = "💰" if s["namespace"] == "earning-reports" else "📄"
                    badges_html += f"<span class='badge {ns_class}'>{ns_icon} [{s['index']}] {s['source']} (p.{s['page']})</span>"
                st.markdown(badges_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Show child vs parent retrieved segments
                st.markdown("### Retrieved Context Segments")
                for s in result["sources"]:
                    with st.expander(f"[{s['index']}] {s['source']} — Page {s['page']} ({s['namespace']})"):
                        sc1, sc2 = st.columns(2)
                        with sc1:
                            st.markdown("**Retrieved Child Node (Dense Match)**")
                            st.caption(f"*{s['content']}*")
                        with sc2:
                            st.markdown("**Fed Parent Segment (Synthesised Context)**")
                            st.caption(f"*{s['parent_content']}...*")

            # Display Pipeline Execution Flow Visualizer inside an expander below results
            st.divider()
            if rag_mode == "Knowledge Graph RAG (Neo4j)":
                with st.expander("🛠️ View Pipeline Execution Flow Visualizer", expanded=False):
                    st.markdown(f"""
                    <div class='vis-container'>
                         <div class='vis-row'>
                            <div class='vis-node highlight'>User Query</div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>Neo4j Graph Store<br><span style='color: #818CF8; font-size: 0.7rem;'>Read-Only Connection</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>Hybrid Retriever<br><span style='color: #818CF8; font-size: 0.7rem;'>Vector + Keyword + Synonyms</span></div>
                        </div>
                        <div class='vis-row' style='justify-content: center; margin: 0.2rem 0;'>
                            <div class='vis-arrow'>▼</div>
                        </div>
                        <div class='vis-row'>
                            <div class='vis-node highlight'>Local Embeddings<br><span style='color: #818CF8; font-size: 0.7rem;'>bge-small-en-v1.5</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>Subgraph Traversal<br><span style='color: #818CF8; font-size: 0.7rem;'>Entity-Relation Paths</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>LLM Generator<br><span style='color: #818CF8; font-size: 0.7rem;'>gpt-4o-mini Synthesis</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                with st.expander("🛠️ View Pipeline Execution Flow Visualizer", expanded=False):
                    expanded_text = result.get('expanded_query', active_query)
                    if use_expansion:
                        st.info(f"🔮 **Query Expansion Output:** {expanded_text}")
                    st.markdown(f"""
                    <div class='vis-container'>
                         <div class='vis-row'>
                            <div class='vis-node highlight'>User Query</div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>Domain Classifier<br><span style='color: #818CF8; font-size: 0.7rem;'>IN_DOMAIN ✓</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node {"highlight" if use_expansion else ""}'>Query Expansion Node<br><span style='color: #818CF8; font-size: 0.7rem;'>{"Mistral Active" if use_expansion else "Bypassed"}</span></div>
                        </div>
                        <div class='vis-row' style='justify-content: center; margin: 0.2rem 0;'>
                            <div class='vis-arrow'>▼</div>
                        </div>
                        <div class='vis-row'>
                            <div class='vis-node highlight'>Retrieval Namespace Filter<br><span style='color: #818CF8; font-size: 0.7rem;'>Mode: '{namespace_mode}'</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>Dense (Vector Search)<br><span style='color: #818CF8; font-size: 0.7rem;'>k=20 chunks (nomic-embed)</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>Sparse (BM25 Search)<br><span style='color: #818CF8; font-size: 0.7rem;'>k=20 chunks (Okapi)</span></div>
                        </div>
                        <div class='vis-row' style='justify-content: center; margin: 0.2rem 0;'>
                            <div class='vis-arrow'>▼</div>
                        </div>
                        <div class='vis-row'>
                            <div class='vis-node highlight'>Convex Fusion Node<br><span style='color: #818CF8; font-size: 0.7rem;'>α={alpha} score fusion</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>Cross-Encoder Reranker<br><span style='color: #818CF8; font-size: 0.7rem;'>MS-Marco MiniLM (top 3)</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>Context Swapper<br><span style='color: #818CF8; font-size: 0.7rem;'>Child ➔ Parent Text (1500 chars)</span></div>
                            <div class='vis-arrow'>➔</div>
                            <div class='vis-node highlight'>LLM Generator<br><span style='color: #818CF8; font-size: 0.7rem;'>Mistral response synthesis</span></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Benchmarking Tab
with benchmark_tab:
    st.markdown("### 📊 Golden Dataset Evaluation Benchmarks")
    df_eval = load_eval_results()
    
    if df_eval is None:
        st.warning("⚠️ **No evaluation run data found.**")
        st.markdown("""
        To populate this dashboard with live metrics:
        1. Run the local evaluation suite in your terminal:
           ```bash
           python evaluation/evaluate.py
           ```
        2. Refresh this page to view real-time RAG pipeline performance.
        """)
    else:
        # Compute overall stats
        avg_hit = df_eval["hit_rate"].mean()
        avg_key = df_eval["keyword_coverage"].mean()
        avg_faith = df_eval["faithfulness"].mean()
        avg_rel = df_eval["answer_relevance"].mean()
        avg_cprec = df_eval["context_precision"].mean()
        avg_crec = df_eval["context_recall"].mean()
        avg_sim = df_eval["semantic_similarity"].mean()
        
        m_cols = st.columns(7)
        with m_cols[0]:
            st.markdown(f"<div class='metric-box has-tooltip'><div class='tooltip-text'><b>Hit Rate</b><br>The fraction of queries for which the correct source document is retrieved in the top-K chunks.</div><div class='metric-val'>{avg_hit:.2f}</div><div class='metric-lbl'>Hit Rate</div></div>", unsafe_allow_html=True)
        with m_cols[1]:
            st.markdown(f"<div class='metric-box has-tooltip'><div class='tooltip-text'><b>Fact Recall (Keywords)</b><br>The percentage of essential keywords and facts from the reference answer that appear in the generated response.</div><div class='metric-val'>{avg_key:.2%}</div><div class='metric-lbl'>Fact Recall (Keywords)</div></div>", unsafe_allow_html=True)
        with m_cols[2]:
            st.markdown(f"<div class='metric-box has-tooltip'><div class='tooltip-text'><b>Semantic Similarity</b><br>Measures the semantic similarity/overlap (using embeddings) between the generated answer and the reference answer.</div><div class='metric-val'>{avg_sim:.2f}</div><div class='metric-lbl'>Semantic Similarity</div></div>", unsafe_allow_html=True)
        with m_cols[3]:
            st.markdown(f"<div class='metric-box has-tooltip'><div class='tooltip-text'><b>Faithfulness</b><br>Measures if the generated response is strictly grounded in the retrieved context, penalizing hallucinations.</div><div class='metric-val'>{avg_faith:.2f}</div><div class='metric-lbl'>Faithfulness</div></div>", unsafe_allow_html=True)
        with m_cols[4]:
            st.markdown(f"<div class='metric-box has-tooltip'><div class='tooltip-text'><b>Answer Relevance</b><br>Measures how directly the generated response addresses the user's query, penalizing redundant or off-topic information.</div><div class='metric-val'>{avg_rel:.2f}</div><div class='metric-lbl'>Answer Relevance</div></div>", unsafe_allow_html=True)
        with m_cols[5]:
            st.markdown(f"<div class='metric-box has-tooltip'><div class='tooltip-text'><b>Ctx Precision</b><br>Measures if the most relevant chunks are ranked higher in the retrieved context list.</div><div class='metric-val'>{avg_cprec:.2f}</div><div class='metric-lbl'>Ctx Precision</div></div>", unsafe_allow_html=True)
        with m_cols[6]:
            st.markdown(f"<div class='metric-box has-tooltip'><div class='tooltip-text'><b>Ctx Recall</b><br>Measures if the retrieved context contains all the information needed to fully answer the query.</div><div class='metric-val'>{avg_crec:.2f}</div><div class='metric-lbl'>Ctx Recall</div></div>", unsafe_allow_html=True)
            
        st.divider()
        
        # Sliced metrics
        st.markdown("### 📈 Category-Sliced Performance")
        st.markdown("The pipeline's metrics segmented across query complexity types:")
        
        sliced = df_eval.groupby("category").agg({
            "hit_rate": "mean",
            "keyword_coverage": "mean",
            "faithfulness": "mean",
            "answer_relevance": "mean",
            "context_precision": "mean",
            "context_recall": "mean",
            "semantic_similarity": "mean"
        }).reset_index()
        
        # Rename columns for clarity
        sliced.columns = [
            "Category", 
            "Hit Rate (Recall)", 
            "Fact Recall (Keyword Coverage)", 
            "Faithfulness", 
            "Answer Relevance", 
            "Context Precision", 
            "Context Recall", 
            "Semantic Similarity"
        ]
        
        st.dataframe(
            sliced.style.format({
                "Hit Rate (Recall)": "{:.3f}",
                "Fact Recall (Keyword Coverage)": "{:.1%}",
                "Faithfulness": "{:.3f}",
                "Answer Relevance": "{:.3f}",
                "Context Precision": "{:.3f}",
                "Context Recall": "{:.3f}",
                "Semantic Similarity": "{:.3f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Detailed table expander
        with st.expander("🔍 View Raw Evaluation Logs (Per Question)"):
            st.dataframe(df_eval[[
                "question", "category", "hit_rate", "keyword_coverage", "faithfulness", "answer_relevance", "context_precision", "context_recall"
            ]], use_container_width=True)
            
    st.divider()
    
    st.markdown("### Key Improvements Under the Hood")
    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.markdown("""
        #### 1. Neutralizing Semantic Traps
        - **Old system**: An adversarial query like *"How do I build a web scraper to collect agent data?"* retrieved chunks because of the keyword "agent", polluting the context.
        - **New system**: Separate namespaces (`/research` vs `/earning-reports`) restrict the query to the correct corpus domain. Query expansion adds contextual tokens, allowing similarity search to match dense agent framework concepts, and convex fusion ranks keyword overlap correctly.
        """)
    with col_imp2:
        st.markdown("""
        #### 2. Resolving the Context Truncation Dilemma
        - **Old system**: Small chunks split paragraphs in half, leaving out crucial context. Large chunks resulted in diluted embeddings that missed specifics.
        - **New system**: Parent-child indexing searches on dense 400-character segments, but provides the LLM with the full 1500-character parent segment. This keeps accuracy high and generation context complete.
        """)

# About Tab
with about_tab:
    st.markdown("### 🗺️ Engine Architecture Diagram")
    
    st.image("assets/architecture.svg", use_container_width=True)
    
    st.divider()
    
    st.markdown("### Engine Architecture Details")
    
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("""
        #### 1. Ingestion Phase
        - **Layout-Aware Markdown Parsing**: PDF structures, tables, and paragraphs are extracted using `LlamaParse` (API) or `pdfplumber` (local markdown table layout converter).
        - **Directory Namespace Router**: Files inside `data/research` and `data/earning-reports` (or classified by name keywords) are tagged with separate namespace attributes in metadata.
        - **Hierarchical Indexing**: Text is split into overlapping parent chunks of 1500 characters. These are subdivided into child chunks of 400 characters. Only child chunks are vectorized using `nomic-embed-text` and stored in Chroma.
        """)
    with ac2:
        st.markdown("""
        #### 2. Query and Retrieval Phase
        - **Query Expansion Node**: User queries are analyzed by local `Mistral 7B` and expanded to inject relevant terms (e.g. "CAMEL" ➔ "CAMEL role-playing multi-agent communicative framework").
        - **Convex Score combination**: Returns dense scores (Chroma distance, min-max normalized) and sparse scores (BM25 score, min-max normalized). Fuses them:
          $$S_{hybrid} = \\alpha \\cdot S_{dense} + (1 - \\alpha) \\cdot S_{sparse}$$
        - **Context Swapper**: The top 3 reranked child chunks retrieve their respective `parent_content` from metadata. This parent text is fed to `Mistral 7B` for final generation.
        """)