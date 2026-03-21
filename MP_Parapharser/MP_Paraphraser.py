import nltk
from nltk.corpus import wordnet
import random
import string

# Download required data (runs only once)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

def get_synonym(word):
    """Find a good synonym using WordNet"""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    if synonyms:
        return random.choice(list(synonyms))
    return word

def humanize_text(text):
    """Main function that makes AI text sound more human"""
    # Split into sentences
    sentences = nltk.sent_tokenize(text)
    new_sentences = []
    
    for sentence in sentences:
        words = sentence.split()
        new_words = []
        
        for word in words:
            # Clean word for lookup
            clean_word = word.strip(string.punctuation).lower()
            
            # Replace ~40% of longer words with synonyms
            if len(clean_word) > 3 and random.random() < 0.40:
                synonym = get_synonym(clean_word)
                if synonym != clean_word:
                    new_words.append(synonym)
                else:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        new_sentence = ' '.join(new_words)
        
        # Add occasional human transitions (keeps it natural)
        if random.random() < 0.15:
            transitions = ["However, ", "In fact, ", "Interestingly, ", "Moreover, "]
            new_sentence = random.choice(transitions) + new_sentence.lower()
        
        # Fix capitalization and punctuation
        new_sentence = new_sentence.capitalize()
        if not new_sentence.endswith(('.', '!', '?')):
            new_sentence += "."
        
        new_sentences.append(new_sentence)
    
    # Join everything back
    humanized = ' '.join(new_sentences)
    
    # Extra human touch: vary punctuation slightly
    humanized = humanized.replace(". ", ".  ").replace("  ", " ")  # occasional double space feel
    
    return humanized

# === Run the tool ===
print("=== AI Text Humanizer (Open Source) ===")
print("Paste your AI-generated paragraph (~250 words) below:")
ai_text = input("\n>> ")

print("\n⏳ Humanizing your text...")
humanized_version = humanize_text(ai_text)

print("\n✅ HUMANIZED VERSION:\n")
print(humanized_version)
print("\nCopy the text above! You can run this script again anytime.")
