# research_agent.py
import os
import json
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
import time

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# ==========================================
# 1. WEB SEARCH FUNCTIONS
# ==========================================
def web_search(query, max_results=3):
    """Search the web using DuckDuckGo"""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return results
    except Exception as e:
        print(f"⚠️ Search error: {e}")
        return []

def read_url(url):
    """Extract text content from a URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=' ', strip=True)
        # Truncate to a reasonable length (2000 characters)
        return text[:2000]
    except Exception as e:
        print(f"⚠️ Error reading {url}: {e}")
        return ""

# ==========================================
# 2. LLM FUNCTIONS (Using Groq)
# ==========================================
def generate_subquestions(topic, num_questions=3):
    """Break down a complex question into sub-questions"""
    prompt = f"""You are a research assistant. Given the topic: "{topic}", generate {num_questions} specific sub-questions that would help build a comprehensive answer. Return ONLY a JSON list of strings.

Example output format: ["sub-question 1", "sub-question 2", "sub-question 3"]

Sub-questions:"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates research sub-questions. Output ONLY valid JSON."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        # Fallback: extract from response if JSON parsing fails
        content = response.choices[0].message.content
        try:
            # Try to find a list in the response
            import re
            matches = re.findall(r'"([^"]*)"', content)
            return matches[:num_questions] if matches else ["What is " + topic, "Why is " + topic + " important?"]
        except:
            return ["What is " + topic, "Why is " + topic + " important?", "How does " + topic + " work?"]

def summarize_content(content, url, question):
    """Summarize content from a URL relevant to a specific question"""
    if not content or len(content) < 50:
        return "No relevant information found."
    
    prompt = f"""Given the following content from a webpage, summarize the key points that answer this question: "{question}"

Content:
{content[:1500]}

Summary (2-3 sentences):"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes web content concisely."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def synthesize_answer(question, summaries):
    """Combine multiple summaries into a comprehensive answer with citations"""
    # Format the summaries with source links
    formatted_summaries = []
    for i, (summary, url) in enumerate(summaries, 1):
        formatted_summaries.append(f"Source {i}: {summary}\n[URL: {url}]")
    
    context = "\n\n".join(formatted_summaries)
    
    prompt = f"""Based on the following research summaries, provide a comprehensive, well-structured answer to the question: "{question}"

Include citations to specific sources using brackets like [Source 1], [Source 2], etc.

Research Summaries:
{context}

Your Answer:"""

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a research synthesizer. Create a detailed, well-organized answer with clear citations."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.4
    )
    return response.choices[0].message.content

# ==========================================
# 3. MAIN AGENT FUNCTION
# ==========================================
def research_agent(question, top_results=3):
    """Main function - takes a question, searches, and synthesizes an answer"""
    print(f"\n🔍 Researching: {question}")
    print("-" * 50)
    
    # Step 1: Generate sub-questions
    print("📝 Breaking down into sub-questions...")
    sub_questions = generate_subquestions(question)
    for i, sq in enumerate(sub_questions, 1):
        print(f"  {i}. {sq}")
    print()
    
    all_summaries = []
    
    # Step 2: For each sub-question, search and summarize
    for sq in sub_questions:
        print(f"🔎 Searching for: '{sq}'")
        results = web_search(sq, max_results=top_results)
        
        if not results:
            print(f"  ⚠️ No results found for '{sq}'")
            continue
        
        print(f"  📚 Found {len(results)} results, reading content...")
        
        for result in results:
            url = result['href']
            title = result['title']
            snippet = result['body']
            
            # Read the full content
            content = read_url(url)
            if content and len(content) > 100:
                summary = summarize_content(content, url, sq)
                all_summaries.append((summary, url))
                print(f"    ✅ Summarized: {title[:50]}...")
            else:
                # Use the snippet if content can't be read
                all_summaries.append((snippet, url))
                print(f"    📄 Using snippet: {title[:50]}...")
        
        # Rate limit to avoid being blocked
        time.sleep(1)
    
    # Step 3: Synthesize the final answer
    print("\n🔬 Synthesizing final answer...")
    final_answer = synthesize_answer(question, all_summaries)
    
    return final_answer, all_summaries

# ==========================================
# 4. INTERACTIVE SHELL
# ==========================================
def main():
    print("🧠 Research Agent (with web search & synthesis)")
    print("=" * 50)
    print("Type your research question or 'quit' to exit")
    print("-" * 50)
    
    while True:
        question = input("\n📝 Your question: ").strip()
        if question.lower() in ['quit', 'exit']:
            print("Goodbye! 👋")
            break
        
        if not question:
            print("Please enter a question.")
            continue
        
        try:
            answer, sources = research_agent(question)
            print("\n" + "=" * 50)
            print("📝 FINAL ANSWER:")
            print("-" * 50)
            print(answer)
            print("\n📖 Sources:")
            for i, (_, url) in enumerate(sources, 1):
                print(f"  [{i}] {url}")
            print("=" * 50)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()