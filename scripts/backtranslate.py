import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import traceback
import nltk
nltk.download('punkt_tab')
import re
import sys
from config import FINE_TUNED_MODEL_DIR, MODEL_NAME

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Language mapping for cross-translation pairs (expand as needed)
LANG_MAP = {
    "eng_Latn": "nso_Latn",  # English → Sepedi
    "nso_Latn": "tsn_Latn",  # Sepedi → Setswana
    "tsn_Latn": "nso_Latn",  # Setswana → Sepedi
}

LANG_PATTERN = re.compile(r"^([a-z]{3}_[A-Z][a-z]{3})\.(txt|csv)$")

def detect_langs(filename):
    filename_only = Path(filename).name
    match = LANG_PATTERN.match(filename_only)
    if not match:
        sys.exit(
            f" Invalid filename: {filename_only}\n"
            "Expected format: <langCode>.<ext>\n"
            "Example: eng_Latn.txt or nso_Latn.csv"
        )
    src_lang, ext = match.groups()

    if src_lang not in LANG_MAP:
        sys.exit(f"No target language mapping found for '{src_lang}' in LANG_MAP.")

    tgt_lang = LANG_MAP[src_lang]
    print(f" Detected SRC_LANG: {src_lang} → TGT_LANG: {tgt_lang}")
    return src_lang, tgt_lang, ext

def get_latest_checkpoint(model_dir: Path) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    checkpoints = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return model_dir

    latest_checkpoint = max(checkpoints, key=lambda d: int(d.name.split("-")[-1]))
    print(f" Using latest checkpoint: {latest_checkpoint.name}")
    return latest_checkpoint



def load_model_and_tokenizer(model_path: str):
    model_dir = get_latest_checkpoint(FINE_TUNED_MODEL_DIR)
    if model_dir is not None and model_dir.exists():
        print(f"Loading model/tokenizer from local checkpoint: {model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    else:
        print(f"Local checkpoint not found. Loading base model from hub: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model


def translate(texts, source_lang, target_lang, tokenizer, model, batch_size=8, max_length=512):
    results = []
    tokenizer.src_lang = source_lang
    for i in tqdm(range(0, len(texts), batch_size), desc=f"{source_lang} → {target_lang}"):
        batch = texts[i:i + batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
            results.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
        except Exception as e:
            print(f" Error during translation batch {i}-{i+batch_size}: {e}")
            traceback.print_exc()
            results.extend(["[ERROR]"] * len(batch))
    return results

def evaluate_bleu(reference, hypothesis):
    try:
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)
    except Exception as e:
        print(f"BLEU evaluation error: {e}")
        traceback.print_exc()
        return 0.0


def evaluate_meteor(reference, hypothesis):
    try:
        # Tokenize strings into lists of words
        ref_tokens = nltk.word_tokenize(reference)
        hyp_tokens = nltk.word_tokenize(hypothesis)
        # meteor_score expects lists of tokens for both reference and hypothesis
        return meteor_score([ref_tokens], hyp_tokens)
    except Exception as e:
        print(f" METEOR evaluation error: {e}")
        traceback.print_exc()
        return 0.0

def run_backtranslation(input_file, output_file, model_path, limit=None):
    try:
        SRC_LANG, TGT_LANG, ext = detect_langs(input_file)
    except SystemExit:
        return

    try:
        tokenizer, model = load_model_and_tokenizer(model_path)
    except Exception:
        print(" Failed to load model and tokenizer. Exiting.")
        return

    try:
        if ext == "csv":
            df = pd.read_csv(input_file)
            if "src" not in df.columns:
                print(" CSV input must contain a 'src' column.")
                return
            original_texts = df["src"].dropna().astype(str).tolist()
        else:
            with open(input_file, "r", encoding="utf-8") as f:
                original_texts = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input file '{input_file}': {e}")
        traceback.print_exc()
        return

    if limit:
        original_texts = original_texts[:limit]

    print(f"Loaded {len(original_texts)} sentences for back-translation.")

    try:
        # Forward translation
        translated_tgt = translate(original_texts, SRC_LANG, TGT_LANG, tokenizer, model)
        # Back translation
        back_translated_src = translate(translated_tgt, TGT_LANG, SRC_LANG, tokenizer, model)
    except Exception as e:
        print(f" Error during translation: {e}")
        traceback.print_exc()
        return

    bleu_scores = []
    meteor_scores = []

    for orig, back in zip(original_texts, back_translated_src):
        bleu_scores.append(evaluate_bleu(orig, back))
        meteor_scores.append(evaluate_meteor(orig, back))

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    print(f"\nAverage BLEU score: {avg_bleu:.4f}")
    print(f"Average METEOR score: {avg_meteor:.4f}")

    try:
        df_out = pd.DataFrame({
            "original_src": original_texts,
            f"translated_{TGT_LANG}": translated_tgt,
            f"back_translated_{SRC_LANG}": back_translated_src,
            "bleu_score": bleu_scores,
            "meteor_score": meteor_scores
        })
        df_out.to_csv(output_file, index=False)
        print(f" Saved back-translation results to: {output_file}")
    except Exception as e:
        print(f" Error saving results to '{output_file}': {e}")
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Back-translation with auto language detection")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input file (.txt or .csv)")
    parser.add_argument("--output_file", type=str, default="data/backtranslated.csv", help="CSV output file path")
    parser.add_argument("--model_path", type=str, default="facebook/nllb-200-distilled-600M", help="Model or checkpoint directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sentences to process")

    try:
        args = parser.parse_args()
        run_backtranslation(args.input_file, args.output_file, args.model_path, args.limit)
    except Exception as e:
        print(f" Unexpected error: {e}")
        traceback.print_exc()
