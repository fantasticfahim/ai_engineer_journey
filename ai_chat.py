# ai_chat.py
import os
import json
from groq import Groq
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()

# Initialize the Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

def ask_ai(prompt):
    """Send a prompt to the AI and return the response"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
    
def classify_text(text):
    """Classify text and return structured JSON"""
    response = client.chat.completions.create(
        messages =[
            {"role": "system", "content": "You are a text classifier. Output ONLY valid JSON with fields: topic, sentiment, confidence (0-1), key_phrases (list)."},
            {"role": "user", "content": text}
        ],
        model = "llama-3.3-70b-versatile",  # Lower temperature for more deterministic output
        temperature = 0.3
    )
    # Parse and re-return as JSON to ensure it's valid
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"error": "Invalid JSON response", "raw": response.choices[0].message.content}

def main():
    print("🤖 AI Chat Assistant (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        # Check if user wants classification
        if user_input.lower().startswith('classify:'):
            text_to_classify = user_input[9:].strip()  # Remove 'classify:' prefix
            print("AI (Classification): ", end="", flush=True)
            response = classify_text(text_to_classify)
            print(response)
        else:
            print("AI: ", end="", flush=True)
            response = ask_ai(user_input)
            print(response)

if __name__ == "__main__":
    main()