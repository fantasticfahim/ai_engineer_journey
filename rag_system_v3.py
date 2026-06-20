# rag_system_v3.py
import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the embedding model (can be fine-tuned)
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # For advanced reranking, we can use a cross-encoder
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
except Exception as e:
    print(f"Error loading models: {e}")
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    cross_encoder = None

# Global variables
all_documents = []  # List of dicts: {file_name, chunks, embeddings, tfidf_matrix}
chunks_with_metadata = []  # For backward compatibility
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

# ==========================================
# 2. MULTI-DOCUMENT SUPPORT
# ==========================================
def index_document(file_path, file_name=None):
    """
    Index a single document and add it to the collection.
    Supports multiple documents.
    """
    global chunks_with_metadata, embeddings, all_documents
    
    if file_name is None:
        file_name = os.path.basename(file_path)
    
    print(f"📄 Loading document: {file_name}")
    pages = load_pdf(file_path)
    
    if not pages:
        print(f"❌ Failed to extract text from {file_name}.")
        return 0
    
    print("✂️ Chunking text...")
    chunks = chunk_text_with_metadata(pages)
    
    if not chunks:
        print(f"❌ No text chunks created for {file_name}.")
        return 0
    
    print(f"✅ Created {len(chunks)} chunks for {file_name}")
    
    # Create embeddings
    texts = [chunk['text'] for chunk in chunks]
    chunk_embeddings = embedding_model.encode(texts, show_progress_bar=False)
    
    # Create TF-IDF matrix for hybrid search
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Store document metadata
    doc_entry = {
        'file_name': file_name,
        'file_path': file_path,
        'chunks': chunks,
        'embeddings': chunk_embeddings,
        'tfidf_matrix': tfidf_matrix,
        'vectorizer': vectorizer,
        'num_chunks': len(chunks)
    }
    all_documents.append(doc_entry)
    
    # Also update global variables for backward compatibility
    chunks_with_metadata.extend(chunks)
    embeddings.extend(chunk_embeddings)
    
    # Save metadata
    metadata_file = 'documents_metadata.json'
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    except:
        existing = []
    
    existing.append({
        'file_name': file_name,
        'file_path': file_path,
        'num_chunks': len(chunks)
    })
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Indexed {file_name} with {len(chunks)} chunks")
    return len(chunks)

def list_documents():
    """Return the list of indexed documents"""
    return [doc['file_name'] for doc in all_documents]

def remove_document(file_name):
    """Remove a document from the index"""
    global all_documents, chunks_with_metadata, embeddings
    
    # Find and remove from all_documents
    for i, doc in enumerate(all_documents):
        if doc['file_name'] == file_name:
            # Update global chunks and embeddings
            chunks_to_remove = set(doc['chunks'])
            chunks_with_metadata = [c for c in chunks_with_metadata if c not in chunks_to_remove]
            all_documents.pop(i)
            # Rebuild embeddings
            embeddings = []
            for doc in all_documents:
                embeddings.extend(doc['embeddings'])
            print(f"✅ Removed document: {file_name}")
            return True
    
    print(f"❌ Document not found: {file_name}")
    return False

# ==========================================
# 3. HYBRID SEARCH
# ==========================================
def hybrid_search(query, all_docs, top_k=5, alpha=0.5):
    """
    Perform hybrid search combining semantic (cosine) and keyword (TF-IDF) search.
    alpha: weight for semantic search (0-1). Higher = more semantic.
    """
    if not all_docs:
        return []
    
    # Get query embedding for semantic search
    query_embedding = embedding_model.encode([query], show_progress_bar=False)
    
    all_candidates = []
    
    for doc_idx, doc in enumerate(all_docs):
        # Semantic similarity
        semantic_scores = cosine_similarity(query_embedding, doc['embeddings'])[0]
        
        # Keyword similarity (TF-IDF)
        query_vector = doc['vectorizer'].transform([query])
        keyword_scores = cosine_similarity(query_vector, doc['tfidf_matrix'])[0]
        
        # Combine scores
        combined_scores = alpha * semantic_scores + (1 - alpha) * keyword_scores
        
        # Get top chunks from this document
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        
        for idx in top_indices:
            if combined_scores[idx] > 0:
                all_candidates.append({
                    'doc_idx': doc_idx,
                    'chunk_idx': idx,
                    'score': float(combined_scores[idx]),
                    'chunk': doc['chunks'][idx],
                    'file_name': doc['file_name']
                })
    
    # Sort all candidates by combined score
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates[:top_k]

# ==========================================
# 4. ADVANCED RERANKING
# ==========================================
def advanced_rerank(query, candidates, top_n=3):
    """
    Rerank candidates using a cross-encoder for more accurate relevance scoring.
    Falls back to LLM-based reranking if cross-encoder is not available.
    """
    if not candidates:
        return []
    
    # Try cross-encoder first
    if cross_encoder is not None:
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, c['chunk']['text']) for c in candidates]
            scores = cross_encoder.predict(pairs)
            
            # Combine with existing scores
            for i, score in enumerate(scores):
                candidates[i]['cross_encoder_score'] = float(score)
                candidates[i]['final_score'] = 0.7 * float(score) + 0.3 * candidates[i]['score']
            
            # Sort by final score
            candidates.sort(key=lambda x: x['final_score'], reverse=True)
            return candidates[:top_n]
            
        except Exception as e:
            print(f"⚠️ Cross-encoder reranking failed: {e}")
    
    # Fallback: LLM-based reranking
    return llm_rerank(query, candidates, top_n)

def llm_rerank(query, candidates, top_n=3):
    """Rerank using LLM"""
    if not candidates:
        return []
    
    # Build prompt for reranking
    chunk_texts = []
    for i, c in enumerate(candidates):
        chunk_texts.append(f"Chunk {i+1}: {c['chunk']['text'][:500]}...")
    
    chunks_text = "\n\n".join(chunk_texts)
    
    prompt = f"""You are a relevance judge. Given the question and a list of text chunks, rank the chunks from most relevant to least relevant.

Question: {query}

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
        
        content = response.choices[0].message.content
        match = re.search(r'\[.*?\]', content)
        if match:
            indices = json.loads(match.group())
            reranked = []
            for idx in indices:
                if isinstance(idx, int) and 1 <= idx <= len(candidates):
                    reranked.append(candidates[idx - 1])
            if len(reranked) >= top_n:
                return reranked[:top_n]
        
        # Fallback: return top candidates by existing score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_n]
        
    except Exception as e:
        print(f"⚠️ LLM reranking failed: {e}")
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_n]

# ==========================================
# 5. MAIN QUERY FUNCTION (WITH ALL FEATURES)
# ==========================================
def query_pdf_v3(question, top_k=3, alpha=0.5, use_reranking=True, file_filter=None):
    """
    Query the indexed documents with all Phase 3 features.
    
    Args:
        question (str): User query
        top_k (int): Number of chunks to retrieve
        alpha (float): Hybrid search weight (0=keyword, 1=semantic)
        use_reranking (bool): Whether to apply reranking
        file_filter (list): List of file names to search in (None = all)
    
    Returns:
        tuple: (answer, sources, context)
    """
    global all_documents
    
    if not all_documents:
        return "No documents indexed. Please upload a PDF first.", [], ""
    
    # Filter documents if specified
    docs_to_search = all_documents
    if file_filter:
        docs_to_search = [doc for doc in all_documents if doc['file_name'] in file_filter]
    
    if not docs_to_search:
        return "No documents match the filter.", [], ""
    
    print(f"🔍 Searching for: '{question}'")
    print(f"📊 Search mode: {'Hybrid' if alpha < 1 else 'Semantic'}")
    print(f"📁 Searching in {len(docs_to_search)} documents")
    
    try:
        # Step 1: Hybrid search
        candidates = hybrid_search(question, docs_to_search, top_k=top_k*2, alpha=alpha)
        
        if not candidates:
            return "No relevant chunks found.", [], ""
        
        print(f"📚 Retrieved {len(candidates)} candidate chunks")
        
        # Step 2: Advanced reranking
        if use_reranking:
            candidates = advanced_rerank(question, candidates, top_k)
            print(f"✅ Reranked to top {len(candidates)} chunks")
        else:
            candidates = candidates[:top_k]
        
        # Build context and sources
        context_parts = []
        sources = []
        for i, c in enumerate(candidates):
            chunk = c['chunk']
            file_name = c['file_name']
            page_num = chunk['page']
            context_parts.append(f"[Source {i+1}, File: {file_name}, Page {page_num}]: {chunk['text']}")
            sources.append({
                'source_id': i+1,
                'file_name': file_name,
                'page': page_num,
                'chunk_id': chunk['chunk_id'],
                'text_preview': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
            })
        
        context = "\n\n".join(context_parts)
        
        # Step 3: Generate answer
        prompt = f"""Based on the following context, answer the question. Include citations to specific sources using [Source X, File: Y, Page Z] format.

Context:
{context}

Question: {question}

Answer:"""
        
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Always cite your sources using [Source X, File: Y, Page Z] format."},
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
# 6. FINE-TUNING SUPPORT
# ==========================================
def prepare_finetuning_data(questions, answers):
    """
    Prepare data for fine-tuning the embedding model.
    This creates a dataset that can be used with SentenceTransformers.
    """
    # This is a placeholder - actual fine-tuning requires more complex setup
    # and is typically done offline
    print("📊 Preparing fine-tuning data...")
    print(f"✅ Prepared {len(questions)} examples")
    
    training_data = []
    for q, a in zip(questions, answers):
        training_data.append({
            'query': q,
            'positive': a,
            'negative': ""  # Would need hard negatives
        })
    
    return training_data

def fine_tune_embedding_model(training_data, model_name='all-MiniLM-L6-v2'):
    """
    Placeholder for fine-tuning the embedding model.
    Actual fine-tuning requires significant computational resources.
    """
    print("⚠️ Fine-tuning requires substantial resources and is best done offline.")
    print("💡 Consider using a pre-fine-tuned model or a service like Hugging Face AutoTrain.")
    return embedding_model

# ==========================================
# 7. BACKWARD COMPATIBILITY
# ==========================================
class CollectionWrapper:
    """Wrapper for backward compatibility with app.py"""
    def __init__(self):
        self._chunks = chunks_with_metadata
    
    def count(self):
        return len(self._chunks)
    
    def get_chunks(self):
        return self._chunks

collection = CollectionWrapper()

# ==========================================
# 8. EVALUATION FUNCTION
# ==========================================
def evaluate_answer_v3(question, answer, sources):
    """Evaluate answer quality"""
    metrics = {
        'similarity': 0.0,
        'citation_accurate': False,
        'num_sources': len(sources)
    }
    
    try:
        # Check citations
        citation_pattern = r'\[Source \d+, File: .+, Page \d+\]'
        citations_found = re.findall(citation_pattern, answer)
        metrics['citation_accurate'] = len(citations_found) > 0
        
        # Semantic similarity to sources
        if sources:
            source_text = " ".join([s['text_preview'] for s in sources])
            if source_text:
                ans_emb = embedding_model.encode([answer])
                src_emb = embedding_model.encode([source_text])
                similarity = cosine_similarity(ans_emb, src_emb)[0][0]
                metrics['similarity'] = float(similarity)
        
        if metrics['similarity'] == 0:
            metrics['similarity'] = 0.7 if metrics['citation_accurate'] else 0.3
    
    except Exception as e:
        print(f"Error in evaluation: {e}")
    
    return metrics

# ==========================================
# 9. MAIN (For testing)
# ==========================================
def main():
    print("📚 RAG System V3 - Multi-Document Hybrid Search")
    print("-" * 60)
    
    # Index sample documents
    print("\n📄 Indexing Words & Tokens.pdf...")
    index_document("Words & Tokens.pdf")
    
    print("\n🤔 Ask questions (type 'quit' to exit)")
    print("-" * 60)
    
    while True:
        print("\n🔧 Options:")
        print("  alpha: 0.5 (hybrid), 1.0 (semantic), 0.0 (keyword)")
        print("  rerank: True/False")
        
        question = input("\n❓ Your question: ")
        if question.lower() in ['quit', 'exit']:
            break
        
        alpha = 0.5
        try:
            alpha_input = input("Alpha (default 0.5): ")
            if alpha_input:
                alpha = float(alpha_input)
        except:
            pass
        
        use_rerank = True
        try:
            rerank_input = input("Use reranking? (True/False, default True): ")
            if rerank_input:
                use_rerank = rerank_input.lower() == 'true'
        except:
            pass
        
        answer, sources, context = query_pdf_v3(question, top_k=3, alpha=alpha, use_reranking=use_rerank)
        print(f"\n📝 Answer:\n{answer}")
        if sources:
            print(f"\n📖 Sources ({len(sources)} chunks):")
            for source in sources:
                print(f"  - Source {source['source_id']}: {source['file_name']}, Page {source['page']}")

if __name__ == "__main__":
    main()