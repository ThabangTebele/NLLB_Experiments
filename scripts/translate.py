import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODEL_NAME, DEVICE, DATA_DIR, OUTPUT_DIR, MAX_LENGTH

def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
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
        # Return empty list or placeholders so code can continue
        return [""] * len(texts)

def run_translation():
    input_path = os.path.join(DATA_DIR, "combined.csv")
    df = pd.read_csv(input_path)

    tokenizer, model = load_model()
    translations = []

    # Group by source-target language pairs to ensure consistent batches
    grouped = df.groupby(['src_lang', 'tgt_lang'])

    for (src_lang, tgt_lang), group_df in grouped:
        print(f"\nTranslating from {src_lang} to {tgt_lang} ({len(group_df)} sentences)")

        batch_size = 8
        # Use tqdm progress bar for batches
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

    # Append translations back to original dataframe (preserving order)
    df["translated"] = translations

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "baseline_translations.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n[✓] Translations saved to: {output_file}")

if __name__ == "__main__":
    run_translation()
