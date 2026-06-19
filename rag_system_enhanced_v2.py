# rag_system_enhanced_v2.py
import os
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the embedding model
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading model: {e}")
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Global variables
chunks_with_metadata = []
embeddings = []

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ==========================================
# 1. PDF LOADING & CHUNKING
# ==========================================
def load_pdf(file_path):
    """Extract text from a PDF file with page numbers"""
    try:
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({'page': i, 'text': text})
        return pages
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

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

def index_pdf_with_metadata(file_path):
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
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
    
    # Save metadata
    metadata_file = 'chunk_metadata_v2.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Indexing complete! Metadata saved to {metadata_file}")
    return len(chunks_with_metadata)

# ==========================================
# 2. COLLECTION OBJECT (For app.py compatibility)
# ==========================================
class CollectionWrapper:
    """Wrapper to make chunks_with_metadata look like a ChromaDB collection"""
    def __init__(self, chunks):
        self._chunks = chunks
    
    def count(self):
        return len(self._chunks)
    
    def get_chunks(self):
        return self._chunks

# Create collection object for compatibility
collection = CollectionWrapper(chunks_with_metadata)

# ==========================================
# 3. RERANKING FUNCTION
# ==========================================
def rerank_chunks(question, chunks, top_n=3):
    """
    Rerank retrieved chunks using Groq LLM.
    Returns the top_n chunks after reranking.
    """
    if not chunks:
        return []
    
    # Build a prompt for reranking
    chunk_texts = []
    for i, chunk in enumerate(chunks):
        chunk_texts.append(f"Chunk {i+1}: {chunk['text'][:500]}...")
    
    chunks_text = "\n\n".join(chunk_texts)
    
    prompt = f"""You are a relevance judge. Given the question and a list of text chunks, rank the chunks from most relevant to least relevant.

Question: {question}

Chunks:
{chunks_text}

Return ONLY a JSON array of the chunk numbers in order of relevance (most relevant first).
Example: [2, 1, 4, 3]

Relevance ranking:"""
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a relevance judge. Output ONLY valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        # Parse the response
        content = response.choices[0].message.content
        # Find JSON array
        match = re.search(r'\[.*?\]', content)
        if match:
            indices = json.loads(match.group())
            # Convert to 0-based indices and filter valid ones
            reranked = []
            for idx in indices:
                if isinstance(idx, int) and 1 <= idx <= len(chunks):
                    reranked.append(chunks[idx - 1])
            # If we have enough reranked chunks, return them
            if len(reranked) >= top_n:
                return reranked[:top_n]
        
        # Fallback: return original top chunks
        return chunks[:top_n]
    except Exception as e:
        print(f"⚠️ Reranking failed: {e}")
        return chunks[:top_n]

# ==========================================
# 4. MAIN QUERY FUNCTION (WITH RERANKING)
# ==========================================
def query_pdf_enhanced(question, top_k=3):
    """Query the indexed PDF and generate an answer with source tracking and reranking"""
    global chunks_with_metadata, embeddings
    
    if not chunks_with_metadata or len(embeddings) == 0:
        return "No documents indexed. Please run index_pdf_with_metadata first.", [], ""
    
    print(f"🔍 Searching for: '{question}'")
    
    try:
        # Get embedding for question
        question_embedding = embedding_model.encode([question], show_progress_bar=False)
        
        # Compute similarity scores
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        # Get top indices (retrieve more than needed for reranking)
        initial_top_k = top_k * 2
        top_indices = similarities.argsort()[-initial_top_k:][::-1]
        
        # Build list of candidate chunks
        candidate_chunks = []
        for idx in top_indices:
            chunk = chunks_with_metadata[idx]
            candidate_chunks.append({
                'text': chunk['text'],
                'page': chunk['page'],
                'chunk_id': chunk['chunk_id'],
                'similarity': float(similarities[idx])
            })
        
        print(f"📚 Retrieved {len(candidate_chunks)} candidate chunks for reranking")
        
        # ========== RERANKING STEP ==========
        reranked_chunks = rerank_chunks(question, candidate_chunks, top_k)
        print(f"✅ Reranked to top {len(reranked_chunks)} chunks")
        
        # Build context and sources from reranked chunks
        context_parts = []
        sources = []
        for i, chunk in enumerate(reranked_chunks):
            page_num = chunk['page']
            context_parts.append(f"[Source {i+1}, Page {page_num}]: {chunk['text']}")
            sources.append({
                'source_id': i+1,
                'page': page_num,
                'chunk_id': chunk['chunk_id'],
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            })
        
        context = "\n\n".join(context_parts)
        
        # Prepare the prompt with source tracking
        prompt = f"""Based on the following context, answer the question. Include citations to specific sources using [Source X, Page Y] format.

Context:
{context}

Question: {question}

Answer:"""
        
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
        
    except Exception as e:
        return f"Error: {str(e)}", [], ""

# ==========================================
# 5. EVALUATION FUNCTION
# ==========================================
def evaluate_answer(question, answer, ground_truth, sources):
    """
    Evaluate the answer quality using multiple metrics.
    """
    metrics = {
        'similarity': 0.0,
        'citation_accurate': False,
        'contains_answer': False,
        'length': len(answer)
    }
    
    try:
        # 1. Semantic similarity
        if ground_truth and len(ground_truth) > 5:
            # Encode both answers
            ans_emb = embedding_model.encode([answer])
            gt_emb = embedding_model.encode([ground_truth])
            similarity = cosine_similarity(ans_emb, gt_emb)[0][0]
            metrics['similarity'] = float(similarity)
        else:
            # Fallback: use sources as a proxy
            metrics['similarity'] = 0.7 if sources else 0.3
        
        # 2. Citation accuracy
        if sources:
            # Check if citations are present in the answer
            citation_pattern = r'\[Source \d+, Page \d+\]'
            citations_found = re.findall(citation_pattern, answer)
            metrics['citation_accurate'] = len(citations_found) > 0
        
        # 3. Check if answer contains key information
        if ground_truth and len(ground_truth) > 10:
            ans_words = set(answer.lower().split())
            gt_words = set(ground_truth.lower().split())
            overlap = len(ans_words.intersection(gt_words))
            metrics['contains_answer'] = overlap > 3
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
    
    return metrics

# ==========================================
# 6. MAIN (For testing)
# ==========================================
def main():
    print("📚 Enhanced RAG V2 with Reranking and Evaluation")
    print("-" * 60)
    
    # Index the PDF
    print("Indexing Words & Tokens.pdf...")
    num_chunks = index_pdf_with_metadata("Words & Tokens.pdf")
    print(f"✅ Indexed {num_chunks} chunks")
    
    print("\n🤔 Ask questions (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        question = input("\n❓ Your question: ")
        if question.lower() in ['quit', 'exit']:
            break
        
        answer, sources, context = query_pdf_enhanced(question)
        print(f"\n📝 Answer:\n{answer}")
        if sources:
            print(f"\n📖 Sources ({len(sources)} chunks):")
            for source in sources:
                print(f"  - Source {source['source_id']}: Page {source['page']}")
        
        # Show evaluation
        eval_results = evaluate_answer(question, answer, answer, sources)
        print(f"\n📊 Evaluation: Similarity={eval_results['similarity']:.3f}, Citation Accurate={eval_results['citation_accurate']}")

if __name__ == "__main__":
    main()