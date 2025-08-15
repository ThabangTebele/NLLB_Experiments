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
import re
import sys
from config import FINE_TUNED_MODEL_DIR, MODEL_NAME

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LANG_MAP = {"eng_Latn": "nso_Latn", "nso_Latn": "tsn_Latn", "tsn_Latn": "nso_Latn"}
LANG_PATTERN = re.compile(r"^([a-z]{3}_[A-Z][a-z]{3})\.(txt|csv)$")
CHECKPOINT_FILE = "backtranslation_session_checkpoint.txt"

def detect_langs(filename):
    match = LANG_PATTERN.match(Path(filename).name)
    if not match:
        sys.exit(f"Invalid filename: {filename}. Expected <langCode>.<ext>")
    src_lang, ext = match.groups()
    if src_lang not in LANG_MAP:
        sys.exit(f"No target mapping for '{src_lang}' in LANG_MAP.")
    return src_lang, LANG_MAP[src_lang], ext

def get_latest_checkpoint(model_dir: Path) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")
    checkpoints = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return model_dir
    return max(checkpoints, key=lambda d: int(d.name.split("-")[-1]))

def load_model_and_tokenizer(model_path: str):
    model_dir = get_latest_checkpoint(FINE_TUNED_MODEL_DIR)
    if model_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

def translate(texts, src_lang, tgt_lang, tokenizer, model, batch_size=8, max_length=512):
    tokenizer.src_lang = src_lang
    all_results = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"{src_lang} → {tgt_lang}"):
        batch = texts[i:i+batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
            results = tokenizer.batch_decode(translated, skip_special_tokens=True)
        except Exception as e:
            print(f"Error in batch {i}-{i+batch_size}: {e}")
            traceback.print_exc()
            results = ["[ERROR]"] * len(batch)
        all_results.extend(results)
    return all_results

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

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))

def run_backtranslation(input_file, output_file, model_path, limit=None, batch_size=8):
    src_lang, tgt_lang, ext = detect_langs(input_file)
    tokenizer, model = load_model_and_tokenizer(model_path)

    # Load input texts
    if ext == "csv":
        df = pd.read_csv(input_file)
        if "src" not in df.columns:
            print("CSV input must have a 'src' column")
            return
        texts = df["src"].dropna().astype(str).tolist()
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    if limit:
        texts = texts[:limit]

    start_idx = load_checkpoint()
    print(f"Resuming from index {start_idx}")
    texts = texts[start_idx:]

    all_translated = []
    all_back = []
    all_bleu = []
    all_meteor = []

    for i in range(0, len(texts), batch_size):
        batch_src = texts[i:i+batch_size]

        # Forward translation
        batch_translated = translate(batch_src, src_lang, tgt_lang, tokenizer, model, batch_size=batch_size)

        # Back translation
        batch_back = translate(batch_translated, tgt_lang, src_lang, tokenizer, model, batch_size=batch_size)

        # Metrics
        batch_bleu = [evaluate_bleu(orig, back) for orig, back in zip(batch_src, batch_back)]
        batch_meteor = [evaluate_meteor(orig, back) for orig, back in zip(batch_src, batch_back)]

        # Append to running lists
        all_translated.extend(batch_translated)
        all_back.extend(batch_back)
        all_bleu.extend(batch_bleu)
        all_meteor.extend(batch_meteor)

        # Ensure equal length before DataFrame
        min_len = min(len(all_translated), len(all_back), len(all_bleu), len(all_meteor))
        df_out = pd.DataFrame({
            "original_src": texts[:min_len],
            f"translated_{tgt_lang}": all_translated[:min_len],
            f"back_translated_{src_lang}": all_back[:min_len],
            "bleu_score": all_bleu[:min_len],
            "meteor_score": all_meteor[:min_len]
        })

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_out.to_csv(output_file, index=False)
        save_checkpoint(start_idx + min_len)

        print(f"Batch {i//batch_size + 1} saved. Checkpoint updated to index {start_idx + min_len}")

    print(f"Back-translation complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Back-translation with incremental saving & checkpoint")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="data/backtranslated.csv")
    parser.add_argument("--model_path", type=str, default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    run_backtranslation(args.input_file, args.output_file, args.model_path, args.limit, args.batch_size)
