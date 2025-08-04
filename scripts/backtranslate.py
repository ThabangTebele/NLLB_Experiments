import os
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# Load model and tokenizer globally for efficiency
MODEL_NAME = "facebook/nllb-200-distilled-600M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

# Language codes
en = "eng_Latn"
st = "nso_Latn"  # Sepedi

def translate(texts, source_lang, target_lang, batch_size=8, max_length=512):
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"{source_lang} → {target_lang}"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(DEVICE)
        translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])
        results.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    return results

def evaluate_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)

def evaluate_meteor(reference, hypothesis):
    return meteor_score([reference], hypothesis)

def run_backtranslation(input_file, output_file, limit=None):
    with open(input_file, "r", encoding="utf-8") as f:
        original_en = [line.strip() for line in f if line.strip()]

    if limit:
        original_en = original_en[:limit]

    print(f"[✓] Loaded {len(original_en)} English sentences.")

    # EN → ST
    translated_st = translate(original_en, en, st)

    # ST → EN (Back-translate)
    back_translated_en = translate(translated_st, st, en)

    # Evaluate similarity between original and back-translated English
    bleu_scores = []
    meteor_scores = []

    for orig, back in zip(original_en, back_translated_en):
        bleu_scores.append(evaluate_bleu(orig, back))
        meteor_scores.append(evaluate_meteor(orig, back))

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)

    print(f"\n[✓] Average BLEU score (original vs back-translated): {avg_bleu:.4f}")
    print(f"[✓] Average METEOR score (original vs back-translated): {avg_meteor:.4f}")

    # Save all results
    df = pd.DataFrame({
        "original_en": original_en,
        "translated_st": translated_st,
        "back_translated_en": back_translated_en,
        "bleu_score": bleu_scores,
        "meteor_score": meteor_scores
    })

    df.to_csv(output_file, index=False)
    print(f"[✓] Back-translation and evaluation saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Back-translation EN->ST->EN with evaluation")
    parser.add_argument("--input_file", type=str, required=True, help="Path to monolingual English txt file")
    parser.add_argument("--output_file", type=str, default="backtranslated.csv", help="CSV output path")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sentences to process")

    args = parser.parse_args()
    run_backtranslation(args.input_file, args.output_file, args.limit)
