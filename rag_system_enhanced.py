# rag_system_enhanced.py
import os
import json
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import hashlib

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="pdf_docs",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name='all-MiniLM-L6-v2'
    )
)

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
    chunks_with_metadata = []
    
    for page_data in pages:
        page_num = page_data['page']
        text = page_data['text']
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks_with_metadata.append({
                    'text': chunk,
                    'page': page_num,
                    'chunk_id': f"page_{page_num}_chunk_{len(chunks_with_metadata)}"
                })
    
    return chunks_with_metadata

def index_pdf_with_metadata(file_path):
    """Index a PDF file with metadata into ChromaDB"""
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
    
    # Prepare data for ChromaDB
    texts = [chunk['text'] for chunk in chunks_with_metadata]
    ids = [chunk['chunk_id'] for chunk in chunks_with_metadata]
    metadatas = [{'page': chunk['page']} for chunk in chunks_with_metadata]
    
    # Store metadata separately for retrieval
    # Save metadata to a JSON file for reference
    metadata_file = 'chunk_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Storing in vector database with page metadata...")
    collection.add(
        documents=texts,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"✅ Indexing complete! Metadata saved to {metadata_file}")
    return len(chunks_with_metadata)

def query_pdf_enhanced(question, top_k=3):
    """Query the indexed PDF and generate an answer with source tracking"""
    print(f"🔍 Searching for: '{question}'")
    
    # Retrieve relevant chunks with metadata
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    
    # Extract the retrieved documents and metadata
    # IMPORTANT FIX: Handle None values by providing default empty lists
    retrieved_chunks = results['documents'][0] if results['documents'] else []
    retrieved_metadatas = results['metadatas'][0] if results['metadatas'] and results['metadatas'][0] is not None else []
    retrieved_ids = results['ids'][0] if results['ids'] else []
    
    print(f"📚 Retrieved {len(retrieved_chunks)} relevant chunks")
    
    # If no chunks retrieved, return a helpful message
    if not retrieved_chunks:
        return "I couldn't find relevant information in the document to answer your question.", [], ""
    
    # Build context with page references
    context_parts = []
    sources = []
    for i, (chunk, metadata, chunk_id) in enumerate(zip(retrieved_chunks, retrieved_metadatas, retrieved_ids)):
        # IMPORTANT FIX: Handle case where metadata is None
        if metadata is None:
            page_num = 'Unknown'
        else:
            page_num = metadata.get('page', 'Unknown')
            
        context_parts.append(f"[Source {i+1}, Page {page_num}]: {chunk}")
        sources.append({
            'source_id': i+1,
            'page': page_num,
            'chunk_id': chunk_id,
            'text_preview': chunk[:200] + "..." if len(chunk) > 200 else chunk
        })
    
    context = "\n\n".join(context_parts)
    
    # Prepare the prompt with source tracking
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
    print("📚 Enhanced RAG Document Q&A System (with Source Tracking)")
    print("-" * 60)
    print("Features: Source citations, page numbers, and chunk visualization")
    print("-" * 60)
    
    # Check if we already indexed the PDF
    if collection.count() == 0:
        print("No documents indexed. Indexing Words & Tokens.pdf...")
        num_chunks = index_pdf_with_metadata("Words & Tokens.pdf")
        if num_chunks > 0:
            print(f"✅ Indexed {num_chunks} chunks with metadata")
        else:
            print("❌ Indexing failed. Please check your PDF file.")
            return
    else:
        print(f"✅ Found {collection.count()} chunks already indexed")
        # Check if metadata file exists
        if not os.path.exists('chunk_metadata.json'):
            print("⚠️  Metadata file missing. Re-indexing...")
            num_chunks = index_pdf_with_metadata("Words & Tokens.pdf")
            print(f"✅ Re-indexed {num_chunks} chunks with metadata")
    
    print("\n🤔 Ask questions about the document (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        question = input("\n❓ Your question: ").strip()
        if question.lower() in ['quit', 'exit']:
            print("Goodbye! 👋")
            break
        
        if not question:
            print("Please enter a valid question.")
            continue
        
        try:
            answer, sources, context = query_pdf_enhanced(question)
            print(f"\n📝 Answer:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
            if sources:
                print(f"\n📖 Sources ({len(sources)} chunks):")
                for source in sources:
                    preview = source['text_preview'][:50] + "..." if len(source['text_preview']) > 50 else source['text_preview']
                    print(f"  - Source {source['source_id']}: Page {source['page']}, Chunk: {preview}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()