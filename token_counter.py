# token_counter.py
import json
from datetime import datetime

def count_words(text):
    return len(text.split())

def count_characters(text):
    return len(text)

def count_sentences(text):
    count = 0
    for char in text:
        if char in ['.', '!', '?']:
            count += 1
    return count

def estimate_tokens(word_count):
    return int(word_count / 0.75)

def load_history():
    try:
        with open('history.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return []

def save_entry(history, text, stats):
    new_entry = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
        'text': text,
        'words': stats['words'],
        'characters': stats['characters'],
        'sentences': stats['sentences'],
        'estimated_tokens': stats['tokens']
    }

    history.append(new_entry)

    with open('history.json', 'w') as file:
        json.dump(history, file, indent=2)

def main():
    text = input("Enter your text: ")
    
    words = count_words(text)
    chars = count_characters(text)
    sentences = count_sentences(text)
    tokens = estimate_tokens(words)

    stats = {
        'words': words,
        'characters': chars,
        'sentences': sentences,
        'tokens': tokens
    }
    
    print(f"\nStatistics:")
    print(f"- Words: {words}")
    print(f"- Characters: {chars}")
    print(f"- Sentences: {sentences}")
    print(f"- Estimated tokens: ~{tokens}")
    
    history = load_history()
    save_entry(history, text, stats)

    print(f"Saved to history.json. Total entries so far: {len(history)}")

if __name__ == "__main__":
    main()