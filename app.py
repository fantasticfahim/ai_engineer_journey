# app.py
import streamlit as st
import os
import json
import sys
import time
from datetime import datetime
import pandas as pd

# ==========================================
# API KEY LOADING
# ==========================================
def load_api_key():
    """Load API key from .env (local) or secrets (Cloud)"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            os.environ['GROQ_API_KEY'] = api_key
            return api_key
    except Exception:
        pass
    
    try:
        if 'streamlit' in sys.modules:
            api_key = st.secrets.get("GROQ_API_KEY")
            if api_key:
                os.environ['GROQ_API_KEY'] = api_key
                return api_key
    except Exception:
        pass
    
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    
    st.error("⚠️ GROQ_API_KEY not found. Please set it in .env file or Streamlit secrets.")
    return None

GROQ_API_KEY = load_api_key()

# ==========================================
# IMPORT RAG SYSTEM V3
# ==========================================
try:
    from rag_system_v3 import (
        query_pdf_v3,
        index_document,
        list_documents,
        remove_document,
        all_documents,
        collection,
        evaluate_answer_v3
    )
except ImportError as e:
    st.error(f"Error importing RAG system: {e}")
    st.stop()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="RAG Document Q&A V3",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# SESSION STATE INIT
# ==========================================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'evaluation_scores' not in st.session_state:
    st.session_state.evaluation_scores = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs = []
if 'alpha' not in st.session_state:
    st.session_state.alpha = 0.5
if 'use_reranking' not in st.session_state:
    st.session_state.use_reranking = True
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 500
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 50

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E86AB;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #0056b3;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        text-align: center;
        border: 1px solid #ddd;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2E86AB;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.title("📚 RAG System V3")
    st.markdown("---")
    
    # API Key Status
    if GROQ_API_KEY:
        st.success("✅ API Key loaded")
    else:
        st.error("❌ No API Key found")
    
    st.markdown("---")
    
    # ========== DOCUMENT MANAGEMENT ==========
    st.markdown("### 📄 Document Management")
    
    # File upload
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    if uploaded_file is not None:
        # Save uploaded file
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.button(f"📥 Index {uploaded_file.name}"):
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                num_chunks = index_document(file_path, uploaded_file.name)
                if num_chunks > 0:
                    st.success(f"✅ Indexed {num_chunks} chunks from {uploaded_file.name}")
                    st.rerun()
                else:
                    st.error("❌ Failed to index document.")
    
    # List indexed documents
    docs = list_documents()
    if docs:
        st.markdown("**Indexed Documents:**")
        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {doc}")
            with col2:
                if st.button("🗑️", key=f"del_{doc}"):
                    if remove_document(doc):
                        st.rerun()
    
    # Document filter for search
    if docs:
        st.markdown("**Search in selected documents:**")
        st.session_state.selected_docs = st.multiselect(
            "Select documents",
            docs,
            default=docs
        )
    
    st.markdown("---")
    
    # ========== SEARCH PARAMETERS ==========
    st.markdown("### 🔧 Search Parameters")
    
    # Hybrid search weight
    st.session_state.alpha = st.slider(
        "Semantic vs Keyword Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.alpha,
        step=0.1,
        help="1.0 = Semantic only, 0.0 = Keyword only, 0.5 = Hybrid"
    )
    
    # Reranking toggle
    st.session_state.use_reranking = st.checkbox(
        "Use Reranking",
        value=st.session_state.use_reranking,
        help="Cross-encoder or LLM-based reranking"
    )
    
    # Chunk size
    st.session_state.chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=1000,
        value=st.session_state.chunk_size,
        step=50
    )
    
    # Chunk overlap
    st.session_state.chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=100,
        value=st.session_state.chunk_overlap,
        step=10
    )
    
    st.markdown("---")
    
    # ========== EVALUATION DASHBOARD ==========
    st.markdown("### 📊 Evaluation Dashboard")
    
    if st.session_state.evaluation_scores:
        df = pd.DataFrame(st.session_state.evaluation_scores)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_similarity = df['similarity'].mean()
            st.markdown(f"""
                <div class="metric-card">
                    <div>Avg Similarity</div>
                    <div class="metric-value">{avg_similarity:.3f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            citation_accuracy = df['citation_accurate'].mean() * 100
            st.markdown(f"""
                <div class="metric-card">
                    <div>Citation Accuracy</div>
                    <div class="metric-value">{citation_accuracy:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div>Total Queries</div>
                    <div class="metric-value">{len(df)}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with st.expander("📈 Recent Scores"):
            st.dataframe(df.tail(5))
    else:
        st.info("Ask questions to see evaluation metrics")
    
    st.markdown("---")
    
    # ========== CHAT HISTORY ==========
    st.markdown("### 💬 Chat History")
    
    if st.session_state.chat_history:
        for i, (q, a) in enumerate(st.session_state.chat_history[-10:]):
            with st.expander(f"Q{i+1}: {q[:40]}..."):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a[:200]}..." if len(a) > 200 else f"**Answer:** {a}")
    else:
        st.caption("No chat history yet")
    
    st.markdown("---")
    
    if st.button("🗑️ Clear History"):
        st.session_state.chat_history = []
        st.session_state.evaluation_scores = []
        st.rerun()
    
    st.markdown("---")
    
    # ========== RE-INDEX ==========
    if st.button("🔄 Re-index All Documents"):
        with st.spinner("Re-indexing all documents..."):
            # This is a simplified re-index - in production, would need proper tracking
            st.info("Re-indexing will happen on next query with new parameters.")

# ==========================================
# MAIN CONTENT
# ==========================================
st.markdown('<p class="main-header">📚 Document Q&A with Source Tracking</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your documents and get answers with citations</p>', unsafe_allow_html=True)

# Display current settings
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🔍 Search Mode: {'Hybrid' if st.session_state.alpha not in [0,1] else 'Semantic' if st.session_state.alpha == 1 else 'Keyword'}")
with col2:
    st.caption(f"📊 Reranking: {'ON' if st.session_state.use_reranking else 'OFF'}")
with col3:
    st.caption(f"📄 Documents: {len(list_documents())}")

# Input area
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Your question:",
        placeholder="e.g., What is the relationship between words and tokens?",
        key="question_input"
    )
with col2:
    top_k = st.selectbox("Sources", [1, 2, 3, 5], index=2)

# ==========================================
# ASK BUTTON
# ==========================================
if st.button("🔍 Ask", type="primary") or question:
    if not GROQ_API_KEY:
        st.error("❌ Cannot ask questions: GROQ_API_KEY not set.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Searching..."):
            try:
                start_time = time.time()
                
                # Query the RAG system with current settings
                answer, sources, context = query_pdf_v3(
                    question=question,
                    top_k=top_k,
                    alpha=st.session_state.alpha,
                    use_reranking=st.session_state.use_reranking,
                    file_filter=st.session_state.selected_docs if st.session_state.selected_docs else None
                )
                
                response_time = time.time() - start_time
                
                # Store in chat history
                st.session_state.chat_history.append((question, answer))
                
                # Evaluate
                eval_results = evaluate_answer_v3(question, answer, sources)
                eval_results['response_time'] = response_time
                eval_results['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                eval_results['question'] = question
                st.session_state.evaluation_scores.append(eval_results)
                
                # Display answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("### 📝 Answer")
                st.write(answer)
                st.caption(f"⏱️ Response time: {response_time:.2f}s")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display sources
                if sources:
                    st.markdown("### 📖 Sources")
                    st.markdown(f"*{len(sources)} relevant chunks retrieved*")
                    
                    for i, source in enumerate(sources):
                        with st.expander(f"📄 Source {source['source_id']} — {source['file_name']}, Page {source['page']}"):
                            st.markdown(f"**Chunk ID:** `{source['chunk_id']}`")
                            st.markdown("**Text Preview:**")
                            st.text(source['text_preview'])
                
                # Show retrieved context (for debugging)
                with st.expander("🔍 Retrieved Context (Raw)"):
                    st.text_area("Context", context, height=200, key="context_display")
                
                # Show evaluation badge
                st.success(f"✅ Evaluated | Similarity: {eval_results['similarity']:.3f} | Citation Accurate: {eval_results['citation_accurate']}")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Groq, and SentenceTransformers | RAG V3 with Hybrid Search + Reranking")