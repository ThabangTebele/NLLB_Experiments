import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    DEVICE,
    DATA_DIR,
    PROCESSED_DIR,
    MAX_LENGTH
)

# Fallback for fine-tuned model path
try:
    from config import FINE_TUNED_MODEL_DIR
except ImportError:
    FINE_TUNED_MODEL_DIR = Path("model_finetuned")  # Default if not in config


def get_latest_checkpoint(model_dir: Path) -> Path:
    """
    Find the most recent checkpoint directory inside model_dir.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    # List subdirectories matching checkpoint naming pattern
    checkpoints = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return model_dir  # Assume the model is saved directly here

    # Sort by checkpoint number
    latest = sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))[-1]
    return latest


def load_model():
    try:
        model_path = get_latest_checkpoint(FINE_TUNED_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(DEVICE)
        print(f"[✓] Loaded model from: {model_path}")
        return tokenizer, model
    except Exception as e:
        print(f"[Error] Failed to load model/tokenizer: {e}")
        raise


def translate_texts(texts, src_lang_code, tgt_lang_code, tokenizer, model):
    try:
        tokenizer.src_lang = src_lang_code
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)

        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
            max_length=MAX_LENGTH
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"[Error] Translation failed for batch: {e}")
        return [""] * len(texts)

def run_translation():
    input_path = Path(DATA_DIR) / "combined.csv"
    df = pd.read_csv(input_path)

    tokenizer, model = load_model()
    translations = []

    grouped = df.groupby(['src_lang', 'tgt_lang'])

    for (src_lang, tgt_lang), group_df in grouped:
        print(f"\nTranslating from {src_lang} to {tgt_lang} ({len(group_df)} sentences)")

        batch_size = 8
        for i in tqdm(range(0, len(group_df), batch_size), desc=f"Batch translating {src_lang}→{tgt_lang}"):
            batch = group_df.iloc[i:i+batch_size]
            translated_batch = translate_texts(
                batch["src"].tolist(),
                src_lang_code=src_lang,
                tgt_lang_code=tgt_lang,
                tokenizer=tokenizer,
                model=model
            )
            translations.extend(translated_batch)

    df["translated"] = translations

    output_file = Path(PROCESSED_DIR) / "baseline_translations.csv"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n[✓] Translations saved to: {output_file}")

if __name__ == "__main__":
    run_translation()
