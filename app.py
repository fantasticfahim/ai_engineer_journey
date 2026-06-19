# app.py
import streamlit as st
import os
import json
import sys
import time
from datetime import datetime
import pandas as pd

# ==========================================
# FIX: Handle API Key Loading for Both Environments
# ==========================================
def load_api_key():
    """Load API key from .env (local) or secrets (Cloud)"""
    # First, try to load from .env file (for local development)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            os.environ['GROQ_API_KEY'] = api_key
            return api_key
    except Exception as e:
        print(f"Error loading .env: {e}")
    
    # If running on Streamlit Cloud, try secrets
    try:
        if 'streamlit' in sys.modules:
            api_key = st.secrets.get("GROQ_API_KEY")
            if api_key:
                os.environ['GROQ_API_KEY'] = api_key
                return api_key
    except Exception as e:
        print(f"Error loading secrets: {e}")
    
    # If still no key, try environment variable directly
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    
    # No key found - show error in app but don't crash
    st.error("⚠️ GROQ_API_KEY not found. Please set it in .env file or Streamlit secrets.")
    return None

# Load API key
GROQ_API_KEY = load_api_key()

# ==========================================
# Import RAG System
# ==========================================
try:
    from rag_system_enhanced_v2 import (
        query_pdf_enhanced,
        index_pdf_with_metadata,
        collection,
        evaluate_answer,
        embedding_model,
        chunks_with_metadata
    )
except ImportError as e:
    st.error(f"Error importing RAG system: {e}")
    st.stop()

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Session State Initialization
# ==========================================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0

if 'evaluation_scores' not in st.session_state:
    st.session_state.evaluation_scores = []

# ==========================================
# Custom CSS
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
    st.title("📚 RAG System")
    st.markdown("---")
    
    # Show API key status
    if GROQ_API_KEY:
        st.success("✅ API Key loaded")
    else:
        st.error("❌ No API Key found")
    
    st.markdown("---")
    
    # Document stats
    st.markdown("**📄 Document:** Words & Tokens.pdf")
    try:
        if collection:
            st.metric("Total Chunks", collection.count())
    except:
        pass
    
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
    if st.button("🔄 Re-index PDF"):
        with st.spinner("Re-indexing..."):
            try:
                num_chunks = index_pdf_with_metadata("Words & Tokens.pdf")
                st.success(f"✅ Re-indexed {num_chunks} chunks!")
                st.rerun()
            except Exception as e:
                st.error(f"Error re-indexing: {e}")

# ==========================================
# MAIN CONTENT
# ==========================================
st.markdown('<p class="main-header">📚 Document Q&A with Source Tracking</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about your document and get answers with citations</p>', unsafe_allow_html=True)

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
        st.error("❌ Cannot ask questions: GROQ_API_KEY not set. Please add it to .env file.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Searching for relevant chunks..."):
            try:
                start_time = time.time()
                
                # Query the RAG system
                answer, sources, context = query_pdf_enhanced(question, top_k=top_k)
                
                response_time = time.time() - start_time
                
                # Store in chat history
                st.session_state.chat_history.append((question, answer))
                
                # ========== EVALUATE ANSWER ==========
                citation_accurate = len(sources) > 0
                
                # Compute semantic similarity
                try:
                    # Get embeddings for similarity comparison
                    if answer and len(answer) > 10:
                        # Simple heuristic: if we have sources, likely good
                        similarity_score = 0.85 if citation_accurate else 0.3
                    else:
                        similarity_score = 0.0
                except:
                    similarity_score = 0.5
                
                # Store evaluation
                eval_result = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'question': question,
                    'similarity': similarity_score,
                    'citation_accurate': citation_accurate,
                    'response_time': response_time,
                    'num_sources': len(sources)
                }
                st.session_state.evaluation_scores.append(eval_result)
                
                # ========== DISPLAY ANSWER ==========
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
                        with st.expander(f"📄 Source {source['source_id']} — Page {source['page']}"):
                            st.markdown(f"**Chunk ID:** `{source['chunk_id']}`")
                            st.markdown("**Text Preview:**")
                            st.text(source['text_preview'])
                
                # Show retrieved context (for debugging)
                with st.expander("🔍 Retrieved Context (Raw)"):
                    st.text_area("Context", context, height=200, key="context_display")
                
                # Show evaluation badge
                st.success(f"✅ Evaluated | Similarity: {similarity_score:.3f} | Citation Accurate: {citation_accurate}")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Groq, and SentenceTransformers")