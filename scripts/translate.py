import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import DEVICE, DATA_DIR, PROCESSED_DIR, MAX_LENGTH, BATCH_SIZE, FINE_TUNED_MODEL_DIR, MODEL_NAME

CHECKPOINT_FILE = Path(PROCESSED_DIR) / "translate_checkpoint.txt"
OUTPUT_FILE = Path(PROCESSED_DIR) / "baseline_translations.csv"

def load_model():
    """Load latest fine-tuned model or fallback to base pretrained model."""
    def get_latest_checkpoint(model_dir: Path) -> Path | None:
        if not model_dir.exists():
            return None
        checkpoints = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
        if not checkpoints:
            return model_dir if model_dir.exists() else None
        return sorted(checkpoints, key=lambda x: int(x.name.split("-")[-1]))[-1]

    try:
        model_path = get_latest_checkpoint(Path(FINE_TUNED_MODEL_DIR)) or MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path)).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f"Failed to load model/tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
        return tokenizer, model

def translate_texts(texts, src_lang_code, tgt_lang_code, tokenizer, model):
    """Translate a batch of texts."""
    try:
        tokenizer.src_lang = src_lang_code
        encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code),
            max_length=MAX_LENGTH
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"[Error] Translation failed for batch: {e}")
        return [""] * len(texts)

def load_checkpoint():
    """Load last processed index if checkpoint exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(index):
    """Save last processed index to checkpoint."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))

def run_translation():
    df = pd.read_csv(Path(DATA_DIR) / "combined.csv")
    tokenizer, model = load_model()

    # Check if previous translation exists
    if OUTPUT_FILE.exists():
        saved_df = pd.read_csv(OUTPUT_FILE)
        translations = saved_df["tgt"].tolist()
        start_idx = len(translations)
        print(f"Resuming from index {start_idx}")
    else:
        translations = []
        start_idx = 0

    grouped = df.groupby(['src_lang', 'tgt_lang'])
    total_sentences = len(df)

    for (src_lang, tgt_lang), group_df in grouped:
        indices = group_df.index.tolist()
        # Skip already translated sentences
        indices = [i for i in indices if i >= start_idx]

        for i in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"Translating {src_lang}→{tgt_lang}"):
            batch_indices = indices[i:i+BATCH_SIZE]
            batch_texts = df.loc[batch_indices, "src"].tolist()
            translated_batch = translate_texts(batch_texts, src_lang, tgt_lang, tokenizer, model)
            translations.extend(translated_batch)

            # Save progress incrementally
            df.loc[:len(translations)-1, "tgt"] = translations
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
            save_checkpoint(len(translations))
            print(f"Saved {len(translations)}/{total_sentences} translations.")

    print(f"\nAll translations complete. Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_translation()
