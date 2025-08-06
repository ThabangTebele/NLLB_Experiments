# config.py
import torch

from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === File paths ===
DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed")
MODEL_DIR = Path("model")
FINE_TUNED_MODEL_DIR = MODEL_DIR / "model_finetuned"

# === Language codes (NLLB format) ===
LANG_CODES = {
    "english": "eng_Latn",
    "sepedi": "nso_Latn",   # Northern Sotho (Sepedi)
    "tswana": "tsn_Latn"
}

# === Optional: Language Pairs for training/testing ===
LANG_PAIRS = [
    (LANG_CODES["english"], LANG_CODES["sepedi"]),
    (LANG_CODES["english"], LANG_CODES["tswana"]),
    # Add more pairs if needed
]

# Default pair (for baseline)
SOURCE_LANG = LANG_CODES["english"]
TARGET_LANG = LANG_CODES["sepedi"]

# === Model ===
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# === Training parameters ===
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# === File names ===
COMBINED_FILE = PROCESSED_DIR / "combined_dataset.csv"
TOKENIZED_DATASET_PATH = PROCESSED_DIR / "tokenized_dataset"
TRANSLATION_OUTPUT = PROCESSED_DIR / "translations.txt"
EVAL_RESULTS_FILE = PROCESSED_DIR / "eval_results.csv"

# === Evaluation ===
METRICS = ["bleu", "sacrebleu","meteor","comet"]
