# app.py
import streamlit as st
import os
import json
import sys

# Check if running on Streamlit Cloud
if 'streamlit' in sys.modules:
    try:
        # Running on Streamlit Cloud - use secrets
        os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]
    except Exception as e:
        st.error(f"Error loading secrets: {e}")

# Import the RAG system
try:
    from rag_system_simple import query_pdf_simple, index_pdf_simple, chunks_with_metadata as collection, embeddings
except ImportError as e:
    st.error(f"Error importing RAG system: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.title("📚 RAG System")
    st.markdown("---")
    st.markdown("**Document:** Words & Tokens.pdf")
    
    # Show document stats
    try:
        if collection:
            st.metric("Total Chunks", len(collection))
    except:
        pass
    
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
    1. Your PDF is split into chunks
    2. Each chunk is embedded and stored
    3. Your question finds relevant chunks
    4. AI generates an answer with sources
    """)
    
    st.markdown("---")
    if st.button("🔄 Re-index PDF"):
        with st.spinner("Re-indexing..."):
            try:
                num_chunks = index_pdf_simple("Words & Tokens.pdf")
                st.success(f"✅ Re-indexed {num_chunks} chunks!")
                st.rerun()
            except Exception as e:
                st.error(f"Error re-indexing: {e}")

# Main content
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

# Ask button
if st.button("🔍 Ask", type="primary") or question:
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Searching for relevant chunks..."):
            try:
                answer, sources, context = query_pdf_simple(question, top_k=top_k)
                
                # Display answer in a styled box
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("### 📝 Answer")
                st.write(answer)
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
                
                # Show retrieved context (for debugging, optional)
                with st.expander("🔍 Retrieved Context (Raw)"):
                    st.text_area("Context", context, height=200, key="context_display")
                    
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, and Groq")