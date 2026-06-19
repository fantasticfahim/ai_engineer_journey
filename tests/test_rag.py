# tests/test_rag.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from rag_system_simple import chunk_text_with_metadata, load_pdf
import tempfile
from pypdf import PdfWriter

def test_chunking():
    dummy_pages = [{'page': 1, 'text': 'word ' * 1000}]
    chunks = chunk_text_with_metadata(dummy_pages, chunk_size=100, overlap=10)
    assert len(chunks) > 0, "Chunking failed to create any chunks."

# You can run this with `pytest tests/` after installing pytest.