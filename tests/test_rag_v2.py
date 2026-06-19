# tests/test_rag_v2.py
import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag_system_enhanced_v2 import (
    chunk_text_with_metadata,
    load_pdf,
    index_pdf_with_metadata,
    query_pdf_enhanced,
    evaluate_answer,
    chunks_with_metadata,
    embeddings
)

def test_chunking():
    """Test that chunking creates the expected number of chunks"""
    dummy_pages = [{'page': 1, 'text': 'word ' * 1000}]
    chunks = chunk_text_with_metadata(dummy_pages, chunk_size=100, overlap=10)
    
    assert len(chunks) > 0, "Chunking failed to create any chunks."
    assert chunks[0]['page'] == 1, "Page number not preserved."

def test_indexing():
    """Test that indexing creates embeddings and metadata"""
    pdf_path = "Words & Tokens.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip("Words & Tokens.pdf not found")
    
    num_chunks = index_pdf_with_metadata(pdf_path)
    assert num_chunks > 0, "Indexing failed to create chunks."
    assert len(chunks_with_metadata) == num_chunks, "Chunks list length mismatch."
    assert len(embeddings) == num_chunks, "Embeddings length mismatch."
    
    # Clean up
    global chunks_with_metadata, embeddings
    chunks_with_metadata = []
    embeddings = []

def test_query():
    """Test that querying returns a valid response structure"""
    pdf_path = "Words & Tokens.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip("Words & Tokens.pdf not found")
    
    index_pdf_with_metadata(pdf_path)
    answer, sources, context = query_pdf_enhanced("What is a token?", top_k=1)
    
    assert isinstance(answer, str), "Answer should be a string."
    assert isinstance(sources, list), "Sources should be a list."
    
    # Clean up
    global chunks_with_metadata, embeddings
    chunks_with_metadata = []
    embeddings = []

def test_evaluation():
    """Test that evaluation metrics work correctly"""
    metrics = evaluate_answer(
        question="What is a token?",
        answer="A token is a unit of text [Source 1, Page 2]",
        ground_truth="A token is a unit of text",
        sources=[{'page': 2, 'chunk_id': 'test'}]
    )
    
    assert 'similarity' in metrics, "Missing similarity metric"
    assert 'citation_accurate' in metrics, "Missing citation_accurate metric"