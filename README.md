#  Exploring Machine Translation Strategies for Sepedi  
### A Study on Low-Resource Neural Machine Translation using NLLB-200

This repository contains the full experimental setup, datasets, and scripts used in the Honours research project:  
**“Exploring Machine Translation Strategies for Sepedi, a Low-Resource South African Language.”**  

It enables complete reproducibility of the experiments conducted on **Sepedi↔English** translation, including **back-translation** and **cross-translation** augmentation, using Meta AI’s **No Language Left Behind (NLLB-200)** model.

---

##  Project Overview

Sepedi (Northern Sotho) is a low-resource South African language with limited bilingual digital text data.  
This research investigates three main strategies to enhance translation quality:

1. **Baseline Evaluation** – Assess the pretrained NLLB-200 model on Sepedi–English translation.  
2. **Back-Translation** – Augment training data by generating synthetic parallel pairs from English monolingual text.  
3. **Cross-Translation** – Leverage Setswana–English data to create Sepedi–English synthetic pairs through linguistic transfer.

All experiments were conducted using the **traditional scientific experimentation methodology**, ensuring controlled replication and measurable outcomes.

---

## Repository Structure
NLLB_Experiments/
│
├── config.py # Global configuration (paths, constants, hyperparameters)
├── preprocess.py # Cleans and aligns parallel corpora
├── train.py # Fine-tunes NLLB-200 model
├── translate.py # Translates and evaluates with BLEU/METEOR
├── backtranslate.py # Generates synthetic data (English → Sepedi → English)
├── requirements.txt # Python dependencies
│
├── data/
│ ├── nso_eng_parallel.csv # Sepedi–English corpus (SADiLaR)
│ ├── tsn_eng_parallel.csv # Setswana–English corpus (SADiLaR)
│ ├── wikipedia_en.txt # English monolingual corpus (TensorFlow Wikipedia 2023)
│ └── processed/ # Cleaned, aligned, and merged outputs
│
├── models/
│ └── nllb_finetuned/ # Model checkpoints and fine-tuned versions
│
└── results/
├── baseline_translations.csv
├── backtranslation_results.csv
├── cross_translation_results.csv
├── evaluation_summary.csv
└── training_logs/


---

##  Environment Setup

### 1. Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate


---
## Install dependencies
pip install -r requirements.txt


## Verify PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

## Configuration (config.py)
MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda"  # or "cpu"
DATA_DIR = "data/"
PROCESSED_DIR = "data/processed/"
FINE_TUNED_MODEL_DIR = "models/nllb_finetuned/"
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3

## Step 1 — Data Preprocessing
python preprocess.py

## Step 2 — Baseline Evaluation
python translate.py --input_file data/processed/combined.csv \
                    --output_file results/baseline_translations.csv


## Step 3 — Back-Translation
python backtranslate.py --input_file data/wikipedia_en.txt \
                        --output_file results/backtranslated.csv \
                        --model_path models/nllb_finetuned/


## Step 4 — Cross-Translation
python translate.py --input_file data/tsn_eng_parallel.csv \
                    --output_file results/cross_translation.csv


