import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import (
    DEVICE,
    DATA_DIR,
    PROCESSED_DIR,
    MAX_LENGTH,
    BATCH_SIZE,
    FINE_TUNED_MODEL_DIR,
    MODEL_NAME
)

# Fallback for fine-tuned model path if not present in config
try:
    from config import FINE_TUNED_MODEL_DIR
except ImportError:
    FINE_TUNED_MODEL_DIR = Path("models/nllb-finetuned")  # Default if not in config

def load_model():
    """
    Loads the latest fine-tuned model checkpoint if available,
    otherwise falls back to the base pretrained model.
    Returns the tokenizer and model.
    """
    def get_latest_checkpoint(model_dir: Path) -> Path | None:
        if not model_dir.exists():
            return None
        checkpoints = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not checkpoints:
            # No checkpoints, but folder exists: treat folder as model path
            return model_dir
        # Return latest checkpoint folder
        return sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))[-1]

    try:
        model_path = get_latest_checkpoint(Path(FINE_TUNED_MODEL_DIR))

        if model_path is None:
            print(f"Fine-tuned model not found at '{FINE_TUNED_MODEL_DIR}'. Loading base pretrained model '{MODEL_NAME}'.")
            model_path = MODEL_NAME
        else:
            print(f" Loading fine-tuned model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(DEVICE)
        return tokenizer, model

    except Exception as e:
        print(f" Failed to load model/tokenizer from '{model_path if 'model_path' in locals() else 'unknown'}': {e}")
        print(f"Falling back to base pretrained model '{MODEL_NAME}'.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
            return tokenizer, model
        except Exception as e2:
            print(f"[Critical Error] Failed to load base pretrained model '{MODEL_NAME}': {e2}")
            raise e2

def translate_texts(texts, src_lang_code, tgt_lang_code, tokenizer, model):
    """
    Translates a list of texts from src_lang_code to tgt_lang_code using the provided model and tokenizer.
    Returns a list of translated strings.
    """
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
    """
    Main workflow for translating all sentences in combined.csv.
    - Loads the combined dataset.
    - Loads the model and tokenizer.
    - Groups by language pairs and translates in batches.
    - Saves the translations to processed/baseline_translations.csv.
    """
    input_path = Path(DATA_DIR) / "combined.csv"
    df = pd.read_csv(input_path)

    tokenizer, model = load_model()
    translations = []

    # Group by language pairs to translate each pair separately
    grouped = df.groupby(['src_lang', 'tgt_lang'])

    for (src_lang, tgt_lang), group_df in grouped:
        print(f"\nTranslating from {src_lang} to {tgt_lang} ({len(group_df)} sentences)")

        batch_size = BATCH_SIZE
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

    # Add translations to the DataFrame
    df["tgt"] = translations

    # Save the results to the processed directory
    output_file = Path(PROCESSED_DIR) / "baseline_translations.csv"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n Translations saved to: {output_file}")

if __name__ == "__main__":
    run_translation()
