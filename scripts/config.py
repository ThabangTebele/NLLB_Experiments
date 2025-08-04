# config.py

from pathlib import Path

# === File paths ===
DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed")
MODEL_DIR = Path("model")

# === Model ===
MODEL_NAME = "facebook/nllb-200-distilled-600M"  # lighter model for faster experimentation
SOURCE_LANG = "eng_Latn"
TARGET_LANG = "nso_Latn"  # Replace with correct language code (e.g., Sepedi = 'nso_Latn')

# === Training parameters ===
MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5

# === File names ===
COMBINED_FILE = PROCESSED_DIR / "combined_dataset.csv"
TOKENIZED_DATASET_PATH = PROCESSED_DIR / "tokenized_dataset"
TRANSLATION_OUTPUT = PROCESSED_DIR / "translations.txt"
EVAL_RESULTS_FILE = PROCESSED_DIR / "eval_results.json"

# === Evaluation ===
METRICS = ["bleu", "sacrebleu"]
