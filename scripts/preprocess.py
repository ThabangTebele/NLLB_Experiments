from pathlib import Path
import pandas as pd
from config import DATA_DIR, LANG_PAIRS
import traceback

def load_and_clean_text(file_path: Path):
    try:
        with file_path.open(encoding="utf-8") as f:
            lines = f.readlines()
        return [line.strip() for line in lines if line.strip()]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        traceback.print_exc()
        return []

def preprocess_datasets(limit = None ):
    combined_data = []
    for src_lang, tgt_lang in LANG_PAIRS:
        src_file = DATA_DIR / f"{src_lang}.txt"
        tgt_file = DATA_DIR / f"{tgt_lang}.txt"

        if not src_file.exists() or not tgt_file.exists():
            print(f" Missing files: {src_file} or {tgt_file}")
            continue

        src_lines = load_and_clean_text(src_file)
        tgt_lines = load_and_clean_text(tgt_file)

        if not src_lines or not tgt_lines:
            print(f"Skipping pair ({src_lang}, {tgt_lang}) due to file read error.")
            continue

        if len(src_lines) != len(tgt_lines):
            print(f" WARNING: Mismatched lines in {src_file} ({len(src_lines)}) and {tgt_file} ({len(tgt_lines)})")
            min_len = min(len(src_lines), len(tgt_lines))
            src_lines = src_lines[:min_len]
            tgt_lines = tgt_lines[:min_len]

        # Apply limit here (if specified)
        if limit:
            src_lines = src_lines[:limit]
            tgt_lines = tgt_lines[:limit]

        for src, tgt in zip(src_lines, tgt_lines):
            combined_data.append({"src": src, "tgt": tgt, "src_lang": src_lang, "tgt_lang": tgt_lang})

    try:
        df = pd.DataFrame(combined_data)
        output_path = DATA_DIR / "combined.csv"
        df.to_csv(output_path, index=False, encoding="utf-8")
        print(f" Combined dataset saved to: {output_path}")
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
        preprocess_datasets(limit=limit)
    except Exception as e:
        print(f" Unexpected error: {e}")
        traceback.print_exc()
