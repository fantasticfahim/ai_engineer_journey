# rag_system_simple.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables (in-memory storage)
chunks_with_metadata = []
embeddings = []

def load_pdf(file_path):
    """Extract text from a PDF file with page numbers"""
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if text and text.strip():
            pages.append({'page': i, 'text': text})
    return pages

def chunk_text_with_metadata(pages, chunk_size=500, overlap=50):
    """Split text into chunks with metadata (page numbers)"""
    chunks = []
    for page_data in pages:
        page_num = page_data['page']
        text = page_data['text']
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append({
                    'text': chunk,
                    'page': page_num,
                    'chunk_id': f"page_{page_num}_chunk_{len(chunks)}"
                })
    return chunks

def index_pdf_simple(file_path):
    """Index a PDF file by creating embeddings"""
    global chunks_with_metadata, embeddings
    
    print(f"📄 Loading PDF: {file_path}")
    pages = load_pdf(file_path)
    
    if not pages:
        print("❌ Failed to extract text from PDF.")
        return 0
    
    print("✂️ Chunking text with metadata...")
    chunks_with_metadata = chunk_text_with_metadata(pages)
    
    if not chunks_with_metadata:
        print("❌ No text chunks created.")
        return 0
    
    print(f"✅ Created {len(chunks_with_metadata)} chunks")
    
    # Create embeddings
    texts = [chunk['text'] for chunk in chunks_with_metadata]
    print("💾 Creating embeddings...")
    embeddings = embedding_model.encode(texts)
    
    # Save metadata
    metadata_file = 'chunk_metadata_simple.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Indexing complete! Metadata saved to {metadata_file}")
    return len(chunks_with_metadata)

def query_pdf_simple(question, top_k=3):
    """Query the indexed PDF and generate an answer"""
    global chunks_with_metadata, embeddings
    
    if not chunks_with_metadata:
        return "No documents indexed. Please run index_pdf_simple first.", [], ""
    
    print(f"🔍 Searching for: '{question}'")
    
    # Get embedding for question
    question_embedding = embedding_model.encode([question])
    
    # Compute similarity scores
    similarities = cosine_similarity(question_embedding, embeddings)[0]
    
    # Get top_k indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Build context and sources
    context_parts = []
    sources = []
    for i, idx in enumerate(top_indices):
        chunk = chunks_with_metadata[idx]
        page_num = chunk['page']
        context_parts.append(f"[Source {i+1}, Page {page_num}]: {chunk['text']}")
        sources.append({
            'source_id': i+1,
            'page': page_num,
            'chunk_id': chunk['chunk_id'],
            'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
        })
    
    context = "\n\n".join(context_parts)
    
    # Prepare prompt with source tracking
    prompt = f"""Based on the following context, answer the question. Include citations to specific sources using [Source X, Page Y] format.

Context:
{context}

Question: {question}

Answer:"""

    # Call Groq API
    from groq import Groq
    from dotenv import load_dotenv
    load_dotenv()
    
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Always cite your sources using [Source X, Page Y] format."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    return answer, sources, context

def main():
    print("📚 Simple RAG Document Q&A System (No ChromaDB)")
    print("-" * 60)
    
    # Index the PDF
    print("Indexing Words & Tokens.pdf...")
    num_chunks = index_pdf_simple("Words & Tokens.pdf")
    print(f"✅ Indexed {num_chunks} chunks")
    
    while True:
        question = input("\n❓ Your question: ")
        if question.lower() in ['quit', 'exit']:
            break
        answer, sources, context = query_pdf_simple(question)
        print(f"\n📝 Answer:\n{answer}")
        print(f"\n📖 Sources ({len(sources)} chunks):")
        for source in sources:
            print(f"  - Source {source['source_id']}: Page {source['page']}")

if __name__ == "__main__":
    main()