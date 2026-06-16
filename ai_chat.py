# ai_chat.py
import os
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
            model="llama-3.3-70b-versatile",  # Free, fast model
            temperature=0.7,  # Controls creativity (0-1)
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

def main():
    print("🤖 AI Chat Assistant (type 'quit' to exit)")
    print("-" * 40)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        print("AI: ", end="", flush=True)
        response = ask_ai(user_input)
        print(response)

if __name__ == "__main__":
    main()