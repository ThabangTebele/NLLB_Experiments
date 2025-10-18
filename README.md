#  Exploring Machine Translation Strategies for Sepedi
### A Study on Low-Resource Neural Machine Translation using NLLB-200

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Research--Only-lightgrey)
![HuggingFace](https://img.shields.io/badge/Transformers-🤗-yellow)
![Status](https://img.shields.io/badge/Status-Research%20Complete-brightgreen)

## Overview
This repository contains all scripts, configurations, and datasets used in the Honours research project:  
**“Exploring Machine Translation Strategies for Sepedi, a Low-Resource South African Language.”**

The project investigates the effectiveness of **Neural Machine Translation (NMT)** approaches for **Sepedi↔English** translation using **Meta AI’s NLLB-200** model.  
It explores three main strategies:
1. **Baseline Evaluation** – Direct fine-tuning using parallel data.
2. **Back-Translation** – Augmenting training data with synthetic pairs generated from English monolingual corpora.
3. **Cross-Translation** – Leveraging linguistically related languages (Setswana) to generate additional data.

All experiments are reproducible using the included scripts.

---

## Repository Structure
```
📁 project-root/
│
├── data/                     # Raw and processed datasets
│   ├── raw/                  # Unprocessed files (SADiLaR + Wikipedia)
│   ├── processed/            # Cleaned and tokenised corpora
│   └── combined.csv          # Unified dataset ready for model input
│
├── scripts/
│   ├── preprocess.py         # Data cleaning and normalisation
│   ├── translate.py          # Handles translation and evaluation
│   ├── train.py              # Fine-tuning NLLB-200 on custom datasets
│   └── config.py             # Global paths and constants
│
├── results/
│   ├── baseline.csv          # Baseline translation results
│   ├── backtranslation.csv   # Results after augmentation
│   ├── loss_plot.png         # Training loss over epochs
│   └── evaluation_metrics.txt
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/sepedi-nmt-research.git
cd sepedi-nmt-research
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

### 3️. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️. Download Required Models
The experiments use **NLLB-200 distilled (600M)** available from Hugging Face:
```bash
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
```

---

## Data Sources
- **SADiLaR Parallel Corpora** (Sepedi–English)
- **Wikipedia 2023 English Dump** (via TensorFlow Datasets)
- **Autshumato Parallel Texts**
All datasets are cleaned, sentence-aligned, and normalised via `preprocess.py`.

---

## Running Experiments

### Baseline Evaluation
Evaluate the pre-trained NLLB-200 model on the parallel Sepedi–English corpus:
```bash
python scripts/translate.py --input_file combined.csv --output_file results/baseline.csv
```

### Back-Translation
Perform English → Sepedi → English back-translation:
```bash
python scripts/backtranslate.py --input_file wikipedia_en.csv --output_file results/backtranslation.csv
```

### Cross-Translation
Translate Setswana → Sepedi to generate new synthetic data:
```bash
python scripts/crosstranslate.py --input_file tswana_en.csv --output_file results/crosstranslation.csv
```

### Fine-Tuning
Retrain model using augmented datasets:
```bash
python scripts/train.py
```

---

##  Evaluation Metrics
- **BLEU** (Papineni et al., 2002)
- **METEOR** (Banerjee & Lavie, 2005)
- **Validation Loss Curves** are logged automatically.
Evaluation scripts can be re-run using:
```bash
python scripts/evaluate.py
```

---

## Reproducibility
All random seeds are fixed.  
Each experiment logs:
- Model configuration
- Tokeniser version
- Dataset size and source
- Checkpoint ID
- Average BLEU and METEOR scores

Training outputs and configurations are automatically saved under `/results/`.


---

## License
This project is licensed for **academic and non-commercial research** purposes only.
