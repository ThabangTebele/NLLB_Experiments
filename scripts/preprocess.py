from pathlib import Path
import pandas as pd
from config import DATA_DIR, LANG_PAIRS,COMBINED_FILE
import traceback
import re

#mapping: source language -> target language
LANG_MAP = {
    "eng_Latn": "nso_Latn",
    "nso_Latn": "tsn_Latn",
    "tsn_Latn": "nso_Latn",
    # add other pairs here...
}

def detect_language_from_filename(filename: str):
    # Assuming filename format is like "eng_Latn.txt"
    match = re.match(r"([a-z]{3}_[A-Za-z]+)\.txt", filename)
    if match:
        return match.group(1)
    return None

def load_and_clean_text(file_path: Path):
    try:
        with file_path.open(encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        traceback.print_exc()
        return []

def preprocess_datasets_auto(limit=None):
    combined_data = []

    # Find all txt files in DATA_DIR
    txt_files = list(DATA_DIR.glob("*.txt"))

    # Map language codes to files
    lang_to_file = {}
    for f in txt_files:
        lang = detect_language_from_filename(f.name)
        if lang:
            lang_to_file[lang] = f

    for src_lang, tgt_lang in LANG_MAP.items():
        src_file = lang_to_file.get(src_lang)
        tgt_file = lang_to_file.get(tgt_lang)

        if not src_file or not tgt_file:
            print(f"Missing file(s) for language pair: {src_lang}, {tgt_lang}")
            continue

        src_lines = load_and_clean_text(src_file)
        tgt_lines = load_and_clean_text(tgt_file)

        if not src_lines or not tgt_lines:
            print(f"Skipping pair ({src_lang}, {tgt_lang}) due to file read error.")
            continue

        if len(src_lines) != len(tgt_lines):
            print(f"WARNING: Mismatched lines in {src_file} ({len(src_lines)}) and {tgt_file} ({len(tgt_lines)})")
            min_len = min(len(src_lines), len(tgt_lines))
            src_lines = src_lines[:min_len]
            tgt_lines = tgt_lines[:min_len]

        if limit:
            src_lines = src_lines[:limit]
            tgt_lines = tgt_lines[:limit]

        for src, tgt in zip(src_lines, tgt_lines):
            combined_data.append({
                "src": src,
                "tgt": tgt,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang
            })

    try:
        df = pd.DataFrame(combined_data)
        output_path = COMBINED_FILE
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"Combined dataset saved to: {output_path}")
    except Exception as e:
        print(f"Error saving combined dataset: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            print(f"Invalid limit argument: {sys.argv[1]}. Using no limit.")
    try:
        preprocess_datasets_auto(limit=limit)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
