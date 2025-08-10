import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import traceback  # For printing stack traces on errors

# Language codes for NLLB model
SRC_LANG = "eng_Latn"      # Source language: English
TGT_LANG = "nso_Latn"      # Target language: Sepedi

# Use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_latest_checkpoint(model_dir: Path) -> Path:
    """
    Returns the latest checkpoint directory in a model directory.
    If no checkpoints are found, returns the base model directory.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory '{model_dir}' not found.")

    checkpoints = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return model_dir  # No checkpoints — return base path

    latest_checkpoint = max(checkpoints, key=lambda d: int(d.name.split("-")[-1]))
    print(f" Using latest checkpoint: {latest_checkpoint.name}")
    return latest_checkpoint

def load_model_and_tokenizer(model_path: str):
    """
    Loads the tokenizer and model from the specified path (checkpoint or base).
    """
    try:
        model_dir = get_latest_checkpoint(Path(model_path))
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
        return tokenizer, model
    except Exception as e:
        print(f" Error loading model/tokenizer from '{model_path}': {e}")
        traceback.print_exc()
        raise

def translate(texts, source_lang, target_lang, tokenizer, model, batch_size=8, max_length=512):
    """
    Translates a list of texts from source_lang to target_lang using the provided model and tokenizer.
    Returns a list of translated strings.
    """
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
    """
    Computes BLEU score for a single reference-hypothesis pair.
    """
    try:
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)
    except Exception as e:
        print(f"BLEU evaluation error: {e}")
        traceback.print_exc()
        return 0.0

def evaluate_meteor(reference, hypothesis):
    """
    Computes METEOR score for a single reference-hypothesis pair.
    """
    try:
        return meteor_score([reference], hypothesis)
    except Exception as e:
        print(f" METEOR evaluation error: {e}")
        traceback.print_exc()
        return 0.0

def run_backtranslation(input_file, output_file, model_path, limit=None):
    """
    Main workflow for back-translation:
    - Loads the model and tokenizer
    - Reads input English sentences
    - Translates EN→ST, then ST→EN (back-translation)
    - Evaluates BLEU and METEOR scores
    - Saves results to CSV
    """
    try:
        tokenizer, model = load_model_and_tokenizer(model_path)
    except Exception:
        print(" Failed to load model and tokenizer. Exiting.")
        return

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            original_en = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading input file '{input_file}': {e}")
        traceback.print_exc()
        return

    if limit:
        original_en = original_en[:limit]

    print(f"Loaded {len(original_en)} English sentences.")

    try:
        # EN → ST (forward translation)
        translated_st = translate(original_en, SRC_LANG, TGT_LANG, tokenizer, model)

        # ST → EN (back-translation)
        back_translated_en = translate(translated_st, TGT_LANG, SRC_LANG, tokenizer, model)
    except Exception as e:
        print(f" Error during translation: {e}")
        traceback.print_exc()
        return

    # Evaluate similarity between original and back-translated English
    bleu_scores = []
    meteor_scores = []

    for orig, back in zip(original_en, back_translated_en):
        bleu_scores.append(evaluate_bleu(orig, back))
        meteor_scores.append(evaluate_meteor(orig, back))

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

    print(f"\nAverage BLEU score: {avg_bleu:.4f}")
    print(f"Average METEOR score: {avg_meteor:.4f}")

    # Save results to CSV
    try:
        df = pd.DataFrame({
            "original_en": original_en,
            "translated_st": translated_st,
            "back_translated_en": back_translated_en,
            "bleu_score": bleu_scores,
            "meteor_score": meteor_scores
        })

        df.to_csv(output_file, index=False)
        print(f" Saved back-translation to {output_file}")
    except Exception as e:
        print(f" Error saving results to '{output_file}': {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command-line arguments for input/output/model/limit
    parser = argparse.ArgumentParser(description="Back-translation EN→ST→EN with evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to monolingual English .txt file")
    parser.add_argument("--output_file", type=str, default="backtranslated.csv", help="CSV output path")
    parser.add_argument("--model_path", type=str, default="facebook/nllb-200-distilled-600M", help="Base or fine-tuned model directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sentences to process")

    try:
        args = parser.parse_args()
        run_backtranslation(args.input_file, args.output_file, args.model_path, args.limit)
    except Exception as e:
        print(f" Unexpected error: {e}")
        traceback.print_exc()
