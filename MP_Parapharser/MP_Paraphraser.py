from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import random
import re

# Multiple models → more diverse, less predictable output (helps perplexity & burstiness)
MODELS = [
    "Vamsi/T5_Paraphrase_Paws",           # solid baseline
    "ramakumar12345/t5_paraphraser",      # different checkpoint
    " tuner007/pegasus_paraphrase",       # Pegasus style (often more natural flow)
]

print("Loading models... (this may take a couple of minutes the first time)")
devices = "cuda" if torch.cuda.is_available() else "cpu"

tokenizers = []
models = []

for name in MODELS:
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForSeq2SeqLM.from_pretrained(name)
    mod = mod.to(devices)
    tokenizers.append(tok)
    models.append(mod)

print(f"Loaded {len(MODELS)} models on {devices}.")

def chunk_text(text: str, max_words: int = 100) -> list:
    """Split long input so we don't truncate badly"""
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def humanize_post(text: str) -> str:
    """
    Lightweight post-processing to mimic human edits:
    - occasional contractions
    - vary connectors
    - light burstiness (short/long sentences)
    - remove some robotic polish
    """
    # Randomly contract some phrases
    contractions = {
        r"\b(it is)\b": "it's", r"\b(It is)\b": "It's",
        r"\b(does not)\b": "doesn't", r"\b(Do not)\b": "Don't",
        r"\b(I am)\b": "I'm", r"\b(we are)\b": "we're",
    }
    for pat, repl in contractions.items():
        if random.random() > 0.6:
            text = re.sub(pat, repl, text)

    # Occasionally swap however/but/though
    text = re.sub(r"\bHowever\b", random.choice(["But", "Yet", "Still", "That said"]), text)

    # Split some long sentences randomly for burstiness
    sentences = re.split(r'(?<=[.!?])\s+', text)
    new_sents = []
    for s in sentences:
        if len(s.split()) > 18 and random.random() > 0.7:
            split_pos = random.randint(6, len(s.split())//2)
            words = s.split()
            new_sents.append(' '.join(words[:split_pos]) + '. ' + ' '.join(words[split_pos:]))
        else:
            new_sents.append(s)
    text = ' '.join(new_sents)

    return text.strip()

def paraphrase_text(
    text: str,
    max_words: int = 600,
    num_return_sequences: int = 6,
    max_length: int = 256
) -> list:
    if not text.strip():
        return []

    # Handle long text by chunking
    chunks = chunk_text(text, max_words=120)
    all_paraphrases = set()  # avoid duplicates

    prefix = random.choice([
        "paraphrase: ",
        "rewrite this naturally: ",
        "say this differently: ",
        "rephrase casually: "
    ])

    for chunk in chunks:
        input_text = prefix + chunk + " </s>"

        for i, (tok, mod) in enumerate(zip(tokenizers, models)):
            try:
                encoding = tok(
                    input_text,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                input_ids = encoding["input_ids"].to(devices)
                attention_mask = encoding["attention_mask"].to(devices)

                # Vary sampling params per model → more diversity
                temps = [0.9, 1.0, 1.15]
                top_ps = [0.92, 0.95, 0.97]

                outputs = mod.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=True,
                    temperature=random.choice(temps),
                    top_p=random.choice(top_ps),
                    top_k=random.choice([50, 80, 120]),
                    repetition_penalty=1.2 + random.uniform(0, 0.3),  # discourage loops
                    num_return_sequences=min(num_return_sequences // len(MODELS) + 1, 4),
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )

                for out in outputs:
                    decoded = tok.decode(
                        out,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    ).strip()

                    # Very basic filter
                    if decoded.lower() != chunk.lower() and len(decoded) > 10:
                        # Apply light humanization
                        humanized = humanize_post(decoded)
                        if humanized not in all_paraphrases:
                            all_paraphrases.add(humanized)

            except Exception as e:
                print(f"Model {MODELS[i]} failed on chunk: {e}")

    results = list(all_paraphrases)[:num_return_sequences * 2]  # keep best-looking ones
    random.shuffle(results)  # mix order so it doesn't look sorted
    return results

if __name__ == "__main__":
    print("\nEnhanced paraphraser — more human-like & harder to flag")
    print("Enter text (up to ~600 words). Type 'quit' to exit.\n")

    while True:
        user_input = input("Your text: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        if len(user_input.split()) > 600:
            print("Text is over 600 words — trimming to first 600.\n")
            user_input = ' '.join(user_input.split()[:600])

        print("\nGenerating variants... (please wait)")
        paraphrases = paraphrase_text(
            user_input,
            num_return_sequences=8,   # aim for 5–7 good ones after filtering
            max_length=384
        )

        if not paraphrases:
            print("Nothing good came out. Try rephrasing your input a bit.\n")
            continue

        print("\nHere are some more natural-sounding rewrites")
        print("→ Edit them yourself, mix parts, add your own voice/examples!\n")

        for i, para in enumerate(paraphrases, 1):
            print(f"{i}) {para}\n")

        print("-" * 70)
