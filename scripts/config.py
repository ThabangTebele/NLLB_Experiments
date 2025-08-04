# config.py

from pathlib import Path

# === File paths ===
DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed")
MODEL_DIR = Path("model")

# === Language codes (NLLB format) ===
LANG_CODES = {
    "english": "eng_Latn",
    "sepedi": "nso_Latn",   # Northern Sotho (Sepedi)
    "tswana": "tsn_Latn"
}

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
