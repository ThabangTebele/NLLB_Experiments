import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import MODEL_NAME, DEVICE, DATA_DIR, OUTPUT_DIR

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model

def translate_texts(texts, src_lang_code, tgt_lang_code, tokenizer, model):
    tokenizer.src_lang = src_lang_code
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
        max_length=512
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

def run_translation():
    input_path = os.path.join(DATA_DIR, "combined.csv")
    df = pd.read_csv(input_path)

    tokenizer, model = load_model()

    translations = []
    batch_size = 8

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        translated_batch = translate_texts(
            batch["src"].tolist(),
            src_lang_code=batch.iloc[0]["src_lang"],
            tgt_lang_code=batch.iloc[0]["tgt_lang"],
            tokenizer=tokenizer,
            model=model
        )
        translations.extend(translated_batch)

    df["translated"] = translations

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "baseline_translations.csv")
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"[✓] Translations saved to: {output_file}")

if __name__ == "__main__":
    run_translation()
