import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import json

# Initialize the embedding model (free, local)
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
    """Extract text from a PDF file"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def index_pdf(file_path):
    """Index a PDF file into ChromaDB"""
    print(f"📄 Loading PDF: {file_path}")
    text = load_pdf(file_path)
    
    print("✂️ Chunking text...")
    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Generate IDs for each chunk
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add to ChromaDB
    print("💾 Storing in vector database...")
    collection.add(
        documents=chunks,
        ids=ids
    )
    print("✅ Indexing complete!")
    return len(chunks)

def query_pdf(question, top_k=3):
    """Query the indexed PDF and generate an answer"""
    print(f"🔍 Searching for: '{question}'")
    
    # Retrieve relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
    
    # Extract the retrieved documents
    retrieved_chunks = results['documents'][0]
    
    # Build context from retrieved chunks
    context = "\n\n".join(retrieved_chunks)
    
    print(f"📚 Retrieved {len(retrieved_chunks)} relevant chunks")
    
    # Prepare the prompt for Groq
    prompt = f"""Based on the following context, answer the question. If the answer is not in the context, say "I don't have enough information to answer this."

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
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    answer = response.choices[0].message.content
    return answer, retrieved_chunks

def main():
    print("📚 RAG Document Q&A System")
    print("-" * 40)
    
    # Check if we already indexed the PDF
    if collection.count() == 0:
        print("No documents indexed. Indexing Words & Tokens.pdf...")
        num_chunks = index_pdf("Words & Tokens.pdf")
        print(f"Indexed {num_chunks} chunks")
    else:
        print(f"✅ Found {collection.count()} chunks already indexed")
    
    print("\n🤔 Ask questions about the document (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        question = input("\n❓ Your question: ")
        if question.lower() == 'quit':
            print("Goodbye!")
            break
        
        try:
            answer, sources = query_pdf(question)
            print(f"\n📝 Answer: {answer}")
            print(f"\n📖 Sources: {len(sources)} relevant chunks retrieved")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()