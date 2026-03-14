# Humanized Paraphraser – Local & Private Text Rewriter

<img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python&logoColor=white" alt="Python version"/>  
<img src="https://img.shields.io/badge/Transformers-4.30%2B-orange?style=flat&logo=huggingface&logoColor=white" alt="Transformers"/>  
<img src="https://img.shields.io/badge/Avoids%20most%20AI%20detectors-🟢-success?style=flat" alt="detector friendly"/>

A **local** paraphrasing tool that produces more human-like text than single-model solutions — useful for people who want higher perplexity / burstiness and fewer obvious AI fingerprints.

## Features

- Uses **multiple models** (T5 + Pegasus variants) → more natural diversity  
- Adjustable temperature, repetition penalty, top-p/k  
- Very basic post-processing (contractions, sentence splitting, word swaps)  
- Handles up to ~600 words  
- Console interface (easy to copy-paste results)

## Installation

```bash
# 1. Clone
git clone https://github.com/YOUR-USERNAME/paraphraser-humanized.git
cd paraphraser-humanized

# 2. (Recommended) Create virtual environment
python -m venv venv
source venv/bin/activate    # Linux/macOS
# or
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install torch transformers
# GPU version (if you have CUDA):
# pip install torch --extra-index-url https://download.pytorch.org/whl/cu121