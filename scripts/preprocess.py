from pathlib import Path
import pandas as pd
from config import DATA_DIR, LANG_PAIRS

def load_and_clean_text(file_path: Path):
    with file_path.open(encoding="utf-8") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]

def preprocess_datasets():
    combined_data = []
    for src_lang, tgt_lang in LANG_PAIRS:
        src_file = DATA_DIR / f"{src_lang}.txt"
        tgt_file = DATA_DIR / f"{tgt_lang}.txt"

        if not src_file.exists() or not tgt_file.exists():
            print(f"Missing files: {src_file} or {tgt_file}")
            continue

        src_lines = load_and_clean_text(src_file)
        tgt_lines = load_and_clean_text(tgt_file)

        if len(src_lines) != len(tgt_lines):
            print(f"WARNING: Mismatched lines in {src_file} and {tgt_file}")
            min_len = min(len(src_lines), len(tgt_lines))
            src_lines = src_lines[:min_len]
            tgt_lines = tgt_lines[:min_len]

        for src, tgt in zip(src_lines, tgt_lines):
            combined_data.append({"src": src, "tgt": tgt, "src_lang": src_lang, "tgt_lang": tgt_lang})

    df = pd.DataFrame(combined_data)
    output_path = DATA_DIR / "combined.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[✓] Combined dataset saved to: {output_path}")

if __name__ == "__main__":
    preprocess_datasets()
