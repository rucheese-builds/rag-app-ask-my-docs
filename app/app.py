import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import load_pipeline, run_pipeline

st.set_page_config(
    page_title="AgentPulse — AI Agent Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'DM Sans', sans-serif;
    background: #F7F8FC;
    color: #1C1E2E;
}

/* ── Remove top padding ── */
[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
    max-width: 1100px;
}
.block-container { padding-top: 0.5rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1C1E2E !important;
}
[data-testid="stSidebar"] > div {
    padding: 1.5rem 1.2rem;
}
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {
    font-size: 0.88rem !important;
    color: #94A3B8 !important;
    line-height: 1.6 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 0.8rem 0 !important;
}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.7rem !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.3rem !important;
    color: #818CF8 !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Header ── */
.ap-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.5rem 0 0.2rem 0;
}
.ap-logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #4F46E5, #818CF8);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.ap-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1C1E2E;
    letter-spacing: -0.02em;
}
.ap-subtitle {
    font-size: 0.85rem;
    color: #64748B;
    margin: 0.2rem 0 1rem 0;
    line-height: 1.5;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #E2E8F0;
    gap: 0; padding: 0;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px;
    padding: 0.5rem 1rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.87rem;
    font-weight: 500;
    color: #64748B;
}
[data-testid="stTabs"] [aria-selected="true"] {
    color: #4F46E5 !important;
    border-bottom-color: #4F46E5 !important;
}

/* ── Domain pills ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    border-radius: 8px !important;
    border: 1.5px solid #E2E8F0 !important;
    background: #ffffff !important;
    color: #374151 !important;
    padding: 0.5rem 0.7rem !important;
    text-align: left !important;
    width: 100% !important;
    white-space: normal !important;
    height: auto !important;
    transition: all 0.15s ease !important;
    line-height: 1.4 !important;
    font-weight: 400 !important;
}
.stButton > button:hover {
    border-color: #4F46E5 !important;
    color: #4F46E5 !important;
    background: #EEF2FF !important;
}
.stButton > button[kind="primary"] {
    background: #4F46E5 !important;
    border-color: #4F46E5 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #4338CA !important;
    border-color: #4338CA !important;
}

/* ── Text input ── */
.stTextInput > div > div > input {
    font-family: 'DM Sans', sans-serif !important;
    background: #ffffff !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    color: #1C1E2E !important;
    font-size: 0.93rem !important;
    padding: 0.65rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #94A3B8 !important; }

/* ── Selected question display ── */
.selected-q {
    background: #EEF2FF;
    border: 1.5px solid #4F46E5;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    font-size: 0.9rem;
    color: #1C1E2E;
    margin-bottom: 0.6rem;
    line-height: 1.4;
}

/* ── Answer card ── */
.answer-card {
    background: #ffffff;
    border: 1.5px solid #E2E8F0;
    border-left: 4px solid #4F46E5;
    border-radius: 0 12px 12px 0;
    padding: 1.4rem 1.6rem;
    margin: 0.8rem 0;
    line-height: 1.8;
    color: #1C1E2E;
    font-size: 0.92rem;
}

/* ── Source badges ── */
.source-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    background: #EEF2FF;
    border: 1px solid #C7D2FE;
    border-radius: 20px;
    padding: 0.18rem 0.65rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.71rem;
    color: #4338CA;
    margin: 0.2rem 0.2rem 0 0;
    font-weight: 500;
}
.source-badge.earnings {
    background: #F0FDF4;
    border-color: #86EFAC;
    color: #166534;
}

/* ── Thinking box ── */
.thinking-box {
    background: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.77rem;
    color: #64748B;
    line-height: 1.7;
}

/* ── Section label ── */
.section-label {
    font-size: 0.71rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin-bottom: 0.6rem;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    margin-bottom: 0.4rem !important;
}

/* ── About/Benchmarking cards ── */
.info-card {
    background: #ffffff;
    border: 1.5px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
}
.info-card h3 {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1C1E2E;
    margin-bottom: 0.4rem;
}
.info-card p {
    font-size: 0.83rem;
    color: #64748B;
    line-height: 1.6;
}

/* ── Metric cards ── */
.metric-card {
    background: #ffffff;
    border: 1.5px solid #E2E8F0;
    border-radius: 12px;
    padding: 1.1rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.9rem;
    font-weight: 600;
    color: #4F46E5;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.metric-val.red { color: #E11D48; }
.metric-lbl {
    font-size: 0.75rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.metric-ctx {
    font-size: 0.7rem;
    color: #94A3B8;
    margin-top: 0.2rem;
}

hr { border-color: #E2E8F0 !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

DOMAINS = {
    "🏗️ Architecture": {
        "papers": ["Internet of Agents", "Internet 3.0", "AgentVerse"],
        "description": "Large-scale agent network architectures",
        "questions": [
            "How does the Internet of Agents differ from traditional multi-agent systems?",
            "What is the DOVIS protocol in Internet 3.0?",
            "How does AgentVerse organize agents into collaborative ecosystems?",
        ]
    },
    "📡 Protocols": {
        "papers": ["AutoGen", "CAMEL", "L2M2", "DyLAN"],
        "description": "Agent communication frameworks",
        "questions": [
            "How does AutoGen handle multi-agent conversation?",
            "What is role-playing in the CAMEL framework?",
            "How does L2M2 orchestrate sub-agents for complex tasks?",
        ]
    },
    "🛠️ Foundations": {
        "papers": ["ReAct", "Scaling LLM", "Scaling Agent Systems"],
        "description": "Core reasoning and scaling research",
        "questions": [
            "How does ReAct combine reasoning and acting in language models?",
            "What are the scaling laws for multi-agent systems?",
            "How does test-time compute scaling improve agent performance?",
        ]
    },
    "📊 Evaluation": {
        "papers": ["AgentBench", "OpenAgents"],
        "description": "Benchmarking and deployment",
        "questions": [
            "What environments does AgentBench use to evaluate LLMs as agents?",
            "How does OpenAgents make language agents accessible to non-experts?",
            "What are the main failure modes found in AgentBench evaluations?",
        ]
    },
    "💰 Earnings": {
        "papers": ["Salesforce", "Microsoft", "Nvidia", "Google", "ServiceNow", "IBM"],
        "description": "Enterprise AI adoption",
        "questions": [
            "How is Salesforce monetizing Agentforce?",
            "What did Microsoft say about AI agents in their earnings call?",
            "How is Nvidia supporting enterprise AI agent infrastructure?",
        ]
    },
}

@st.cache_resource
def initialize_pipeline():
    return load_pipeline()

def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚡ AgentPulse")
        st.markdown("AI Agent Intelligence Platform")
        st.divider()

        st.markdown("### Database")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Papers", "12")
            st.metric("Chunks", "5,308")
        with col2:
            st.metric("Calls", "12")
            st.metric("Pages", "~740")

        st.divider()
        st.markdown("### Pipeline")
        for step in [
            "🔀 BM25 + vector search",
            "⚖️ RRF fusion (k=60)",
            "🎯 Cross-encoder rerank",
            "💬 Mistral generation",
        ]:
            st.markdown(step)

        st.divider()
        st.markdown("### Evaluation")
        st.markdown("Standard: Hit Rate **1.00**")
        st.markdown("Adversarial: Hit Rate **0.50**")
        st.caption("See Benchmarking tab for full results.")

        st.divider()
        st.markdown("[📂 GitHub](https://github.com/rucheese-builds/rag-app-ask-my-docs)")

def render_results(result, query):
    if not result["in_domain"]:
        st.warning(
            "⚠️ This question is outside my document corpus. "
            "Please ask about AI agents, multi-agent systems, or enterprise AI adoption."
        )
        return

    st.success("✅ Retrieved from corpus")

    with st.expander("🔍 Thinking process", expanded=False):
        sources_used = [s['source'] for s in result['sources']]
        st.markdown(f"""<div class='thinking-box'>
Query: {query}<br>
Classification: IN_DOMAIN ✓<br>
Retrieval: BM25 (k=20) + Vector (k=20) → RRF fusion → top 5<br>
Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 → top 3<br>
Sources: {' · '.join(sources_used)}
</div>""", unsafe_allow_html=True)

    st.markdown(
        '<p class="section-label" style="margin-top:1rem">Answer</p>',
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='answer-card'>{result['answer']}</div>",
        unsafe_allow_html=True
    )

    if result["sources"]:
        st.markdown(
            '<p class="section-label" style="margin-top:1rem">Sources</p>',
            unsafe_allow_html=True
        )
        badges = ""
        for s in result["sources"]:
            stype = s.get("source_type", "research")
            cls = "source-badge earnings" if stype == "earnings" else "source-badge"
            icon = "💰" if stype == "earnings" else "📄"
            badges += f"<span class='{cls}'>{icon} [{s['index']}] {s['source']} p.{s['page']}</span>"
        st.markdown(badges, unsafe_allow_html=True)

        st.markdown(
            '<p class="section-label" style="margin-top:1rem">Retrieved chunks</p>',
            unsafe_allow_html=True
        )
        for s in result["sources"]:
            with st.expander(f"[{s['index']}] {s['source']} — page {s['page']}"):
                st.markdown(f"*{s['content']}*")

def render_explorer_tab(vectorstore, bm25, documents, reranker, llm):
    st.markdown(
        '<p class="ap-subtitle">Query 12 frontier research papers and '
        '12 enterprise AI earnings calls. Every answer cites its source.</p>',
        unsafe_allow_html=True
    )

    if "active_domain" not in st.session_state:
        st.session_state.active_domain = None
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""
    if "selected_q_idx" not in st.session_state:
        st.session_state.selected_q_idx = None
    if "run_query" not in st.session_state:
        st.session_state.run_query = False

    st.markdown(
        '<p class="section-label">Knowledge Domains</p>',
        unsafe_allow_html=True
    )

    cols = st.columns(5)
    for i, (domain_key, domain_data) in enumerate(DOMAINS.items()):
        with cols[i]:
            is_active = st.session_state.active_domain == domain_key
            if st.button(
                f"{domain_key}\n{domain_data['description']}",
                key=f"domain_{i}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_domain = domain_key
                st.session_state.pending_query = ""
                st.session_state.selected_q_idx = None
                st.rerun()

    if st.session_state.active_domain:
        domain = st.session_state.active_domain
        data = DOMAINS[domain]

        st.markdown(
            f'<p class="section-label" style="margin-top:1.2rem">'
            f'Suggested questions — {domain}</p>',
            unsafe_allow_html=True
        )
        st.caption(f"Papers: {' · '.join(data['papers'])}")

        q_cols = st.columns(3)
        for i, q in enumerate(data["questions"]):
            with q_cols[i]:
                is_sel = st.session_state.selected_q_idx == i
                if st.button(
                    q,
                    key=f"sq_{domain}_{i}",
                    use_container_width=True,
                    type="primary" if is_sel else "secondary"
                ):
                    st.session_state.pending_query = q
                    st.session_state.selected_q_idx = i
                    st.session_state.run_query = True
                    st.rerun()

    st.divider()
    st.markdown(
        '<p class="section-label">Ask anything</p>',
        unsafe_allow_html=True
    )

    if st.session_state.pending_query:
        st.markdown(
            f"<div class='selected-q'>🔍 <b>Selected:</b> "
            f"{st.session_state.pending_query}</div>",
            unsafe_allow_html=True
        )

    typed_query = st.text_input(
        "Or type your own question",
        value="",
        placeholder="e.g. How do agents communicate in a web of agents?",
        key="typed_input",
        label_visibility="collapsed"
    )

    run_button = st.button(
        "⚡ Search",
        type="primary",
        use_container_width=True
    )

    if run_button or st.session_state.run_query:
        st.session_state.run_query = False
        active_query = typed_query.strip() or st.session_state.pending_query.strip()

        if not active_query:
            st.warning("Please select a question or type your own.")
            return

        with st.spinner("Searching corpus..."):
            result = run_pipeline(
                active_query,
                vectorstore, bm25, documents, reranker, llm
            )

        st.divider()
        render_results(result, active_query)

def render_benchmarking_tab():
    st.markdown(
        '<p class="section-label">Standard evaluation (4 domain questions)</p>',
        unsafe_allow_html=True
    )
    cols = st.columns(3)
    for col, (val, lbl, ctx) in zip(cols, [
        ("1.00", "Hit Rate", "Every query found correct source"),
        ("0.875", "MRR", "Correct source ranked #1"),
        ("0.799", "Semantic Sim.", "Answer vs ground truth"),
    ]):
        with col:
            st.markdown(f"""<div class='metric-card'>
<div class='metric-val'>{val}</div>
<div class='metric-lbl'>{lbl}</div>
<div class='metric-ctx'>{ctx}</div>
</div>""", unsafe_allow_html=True)

    st.markdown(
        '<p class="section-label" style="margin-top:1.2rem">'
        'After adversarial stress test (9 questions incl. semantic traps)</p>',
        unsafe_allow_html=True
    )
    cols2 = st.columns(3)
    for col, (val, lbl, ctx) in zip(cols2, [
        ("0.50", "Hit Rate", "-50% drop"),
        ("0.464", "MRR", "-41% drop"),
        ("0.656", "Semantic Sim.", "-18% drop"),
    ]):
        with col:
            st.markdown(f"""<div class='metric-card'>
<div class='metric-val red'>{val}</div>
<div class='metric-lbl'>{lbl}</div>
<div class='metric-ctx'>{ctx}</div>
</div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<p class="section-label">Methodology</p>',
        unsafe_allow_html=True
    )
    col1, col2 = st.columns(2)
    with col1:
        for title, body in [
            ("Tier 1 — Retrieval metrics",
             "Deterministic metrics with no LLM judge. Hit Rate, MRR, Precision@3, Recall@3."),
            ("Tier 3 — LLM-as-judge (attempted)",
             "RAGAS and TruLens both failed with local models. Require frontier models for reliable scoring."),
        ]:
            st.markdown(
                f"<div class='info-card'><h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True
            )
    with col2:
        for title, body in [
            ("Tier 2 — Semantic similarity",
             "Cosine similarity between answer and ground truth using nomic-embed-text. Free, no LLM needed."),
            ("Stress testing",
             "Added semantic traps, out-of-corpus queries, cross-document synthesis, and precise factual lookups."),
        ]:
            st.markdown(
                f"<div class='info-card'><h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True
            )

def render_about_tab():
    col1, col2 = st.columns([3, 2])
    with col1:
        for title, body in [
            ("What is AgentPulse?",
             "A domain-specific RAG system over 12 Web of Agents research papers and 12 enterprise AI earnings call transcripts. Query both simultaneously and get cited answers tracing every claim to its source."),
            ("Why this corpus?",
             "The unique value is cross-corpus synthesis. Academic papers describe agent architectures formally. Enterprise earnings calls describe the same concepts in business language. AgentPulse bridges both."),
            ("Limitations",
             "Generation uses Mistral 7B via Ollama. Complex multi-hop reasoning works better with frontier models like GPT-4o or Claude. The retrieval pipeline is strong — generation is the current bottleneck."),
        ]:
            st.markdown(
                f"<div class='info-card'><h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True
            )
    with col2:
        for title, body in [
            ("Research Papers (12)",
             "Internet of Agents · Internet 3.0 · AgentVerse · AutoGen · CAMEL · L2M2 · DyLAN · ReAct · Scaling Agent Systems · Scaling LLM Compute · AgentBench · OpenAgents"),
            ("Earnings Calls (12)",
             "Salesforce Q3+Q4 FY26 · Microsoft Q1+Q4 2025 · Nvidia Q3+Q4 2025 · Google Q3+Q4 2025 · ServiceNow Q3+Q4 2025 · IBM Q3+Q4 2025"),
            ("Tech Stack",
             "LangChain · ChromaDB · rank-bm25 · sentence-transformers · Mistral via Ollama · Streamlit"),
        ]:
            st.markdown(
                f"<div class='info-card'><h3>{title}</h3><p>{body}</p></div>",
                unsafe_allow_html=True
            )

def main():
    render_sidebar()

    st.markdown("""
    <div class='ap-header'>
        <div class='ap-logo'>⚡</div>
        <span class='ap-title'>AgentPulse</span>
    </div>
    """, unsafe_allow_html=True)

    vectorstore, bm25, documents, reranker, llm = initialize_pipeline()

    tab1, tab2, tab3 = st.tabs(["🔍 Explorer", "📊 Benchmarking", "ℹ️ About"])

    with tab1:
        render_explorer_tab(vectorstore, bm25, documents, reranker, llm)
    with tab2:
        render_benchmarking_tab()
    with tab3:
        render_about_tab()

if __name__ == "__main__":
    main()