import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import DEVICE, DATA_DIR, PROCESSED_DIR, MAX_LENGTH, BATCH_SIZE, FINE_TUNED_MODEL_DIR, MODEL_NAME

# Output paths
OUTPUT_FILE = Path(PROCESSED_DIR) / "baseline_translations.csv"

# ---------- Utility functions ----------

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
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code),
            max_length=MAX_LENGTH
        )
        return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    except Exception as e:
        print(f"[Error] Translation failed for batch: {e}")
        return [""] * len(texts)

# ---------- Evaluation functions ----------

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

def evaluate_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    try:
        return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)
    except:
        return 0.0

def evaluate_meteor(reference, hypothesis):
    try:
        return meteor_score([nltk.word_tokenize(reference)], nltk.word_tokenize(hypothesis))
    except:
        return 0.0

# ---------- Main translation script ----------

def run_translation():
    df = pd.read_csv(Path(DATA_DIR) / "combined.csv")
    tokenizer, model = load_model()

    # Initialize result columns if missing
    for col in ["translated_text", "bleu_score", "meteor_score"]:
        if col not in df.columns:
            df[col] = ""

    grouped = df.groupby(['src_lang', 'tgt_lang'])
    total_sentences = len(df)

    for (src_lang, tgt_lang), group_df in grouped:
        pair_key = f"{src_lang}->{tgt_lang}"

        # Only process rows where translation is missing
        indices = group_df.index[group_df["translated_text"].isna() | (group_df["translated_text"] == "")].tolist()
        if not indices:
            print(f"Skipping {pair_key}, all sentences already translated.")
            continue

        print(f"Processing {pair_key}: {len(indices)} sentences to translate.")

        for i in tqdm(range(0, len(indices), BATCH_SIZE), desc=f"Translating {pair_key}"):
            batch_indices = indices[i:i+BATCH_SIZE]
            batch_texts = df.loc[batch_indices, "src"].tolist()
            reference_texts = df.loc[batch_indices, "tgt"].tolist()  # Use reference for evaluation

            translated_batch = translate_texts(batch_texts, src_lang, tgt_lang, tokenizer, model)
            batch_bleu = [evaluate_bleu(ref, pred) for ref, pred in zip(reference_texts, translated_batch)]
            batch_meteor = [evaluate_meteor(ref, pred) for ref, pred in zip(reference_texts, translated_batch)]

            # Save results in DataFrame
            df.loc[batch_indices, "translated_text"] = translated_batch
            df.loc[batch_indices, "bleu_score"] = batch_bleu
            df.loc[batch_indices, "meteor_score"] = batch_meteor

            # Save CSV incrementally
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
            print(f"Saved {len(batch_indices)} translations for {pair_key}.")

    # Compute overall averages
    valid_bleu = df["bleu_score"].dropna().astype(float)
    valid_meteor = df["meteor_score"].dropna().astype(float)
    avg_bleu = valid_bleu.mean() if not valid_bleu.empty else 0.0
    avg_meteor = valid_meteor.mean() if not valid_meteor.empty else 0.0

    print(f"\nAll translations complete. Saved to: {OUTPUT_FILE}")
    print(f"Average BLEU score: {avg_bleu:.4f}")
    print(f"Average METEOR score: {avg_meteor:.4f}")

# ---------- Entry point ----------

if __name__ == "__main__":
    run_translation()
