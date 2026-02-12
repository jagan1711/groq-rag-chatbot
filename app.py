"""
Groq RAG Chatbot — Streamlit Application.
A real-time, streaming chatbot with document RAG and web search capabilities.
"""

import logging
import streamlit as st
from pathlib import Path

from config import SUPPORTED_EXTENSIONS
from rag_engine import RAGEngine

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Groq RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global Theme ────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ──────────────────────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.85);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* ── Sidebar Styling ─────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #e2e8f0;
    }

    /* ── Source Badge ─────────────────────────────────────────────────── */
    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 3px 4px 3px 0;
    }

    /* ── Stats Card ──────────────────────────────────────────────────── */
    .stats-card {
        background: rgba(167, 139, 250, 0.1);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
    }
    .stats-card p {
        margin: 0.25rem 0;
        font-size: 0.88rem;
    }

    /* ── File list ───────────────────────────────────────────────────── */
    .file-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 0.3rem 0;
        font-size: 0.85rem;
        color: #e2e8f0;
    }

    /* ── Chat messages refinement ────────────────────────────────────── */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* ── Footer ──────────────────────────────────────────────────────── */
    .footer-text {
        text-align: center;
        color: #94a3b8;
        font-size: 0.75rem;
        padding: 1rem 0 0.5rem 0;
        border-top: 1px solid rgba(148, 163, 184, 0.2);
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    if "rag_engine" not in st.session_state:
        with st.spinner("🔧 Initializing RAG Engine..."):
            st.session_state.rag_engine = RAGEngine()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []


init_session_state()
engine: RAGEngine = st.session_state.rag_engine


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📁 Document Manager")

    # ── File Uploader ────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload files for RAG",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS.keys()],
        accept_multiple_files=True,
        help="Supported: PDF, DOCX, TXT, CSV, JPG, PNG",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.processed_files:
                with st.spinner(f"📄 Processing **{uploaded_file.name}**..."):
                    try:
                        result = engine.ingest_file(
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                        )
                        st.session_state.processed_files.append(uploaded_file.name)
                        st.success(
                            f"✅ **{uploaded_file.name}** — "
                            f"{result['chunks_stored']} chunks indexed"
                        )
                        if result.get("vision_analysis"):
                            with st.expander("🖼️ Vision Analysis"):
                                st.write(result["vision_analysis"])
                    except Exception as e:
                        st.error(f"❌ Failed: {uploaded_file.name} — {e}")

    # ── Indexed Documents ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📚 Indexed Documents")

    sources = engine.get_sources()
    if sources:
        for source in sources:
            st.markdown(
                f'<div class="file-item">📄 {source}</div>',
                unsafe_allow_html=True,
            )

        # Stats
        st.markdown(
            f"""<div class="stats-card">
            <p>📊 <strong>{engine.document_count}</strong> chunks indexed</p>
            <p>📁 <strong>{len(sources)}</strong> documents loaded</p>
            <p>💬 <strong>{engine.memory.message_count}</strong> messages in memory</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.info("No documents uploaded yet. Upload files above to get started!")

    # ── Actions ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Actions")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Docs", use_container_width=True):
            engine.clear_all()
            st.session_state.processed_files = []
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💬 New Chat", use_container_width=True):
            engine.memory.clear()
            st.session_state.messages = []
            st.rerun()

    # ── Capabilities ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Capabilities")
    st.markdown("""
    - 📄 **PDF, DOCX, TXT, CSV** support
    - 🖼️ **Image OCR + Vision AI**
    - 🌐 **Real-time web search**
    - ⚡ **Streaming responses**
    - 🧠 **Conversation memory**
    - 📎 **Source citations**
    """)


# ─── Main Chat Area ─────────────────────────────────────────────────────────
st.markdown(
    """<div class="main-header">
        <h1>⚡ Groq RAG Chatbot</h1>
        <p>Powered by Groq • ChromaDB • Tavily | Upload docs, search the web, get answers</p>
    </div>""",
    unsafe_allow_html=True,
)

# ── Display Chat History ─────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# ── Chat Input ───────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask anything — about your docs or the web..."):
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Stream assistant response
    with st.chat_message("assistant", avatar="🤖"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for token in engine.query(user_input):
                full_response += token
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ An error occurred: {str(e)}"
            response_placeholder.markdown(full_response)
            logger.error(f"Chat error: {e}")

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

# ── Footer ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer-text">'
    "Built with ❤️ using Groq • LLaMA 3.3 • ChromaDB • Tavily • Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
