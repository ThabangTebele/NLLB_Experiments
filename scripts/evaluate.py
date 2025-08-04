import os
import pandas as pd
from config import OUTPUT_DIR, EVAL_METRICS

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sacrebleu.metrics import TER

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

def evaluate_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)

def evaluate_meteor(reference, hypothesis):
    return meteor_score([reference], hypothesis)

def evaluate_ter(reference, hypothesis):
    return TER().corpus_score([hypothesis], [[reference]]).score

def evaluate_comet(df):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    samples = [
        {"src": row["src"], "mt": row["translated"], "ref": row["tgt"]}
        for _, row in df.iterrows()
    ]
    scores = model.predict(samples, batch_size=8, gpus=1 if COMET_AVAILABLE else 0)
    return sum(scores.scores) / len(scores.scores)

def run_evaluation():
    input_path = os.path.join(OUTPUT_DIR, "baseline_translations.csv")
    df = pd.read_csv(input_path)

    print(f"[✓] Evaluating {len(df)} translations...\n")
    results = {}

    if "bleu" in EVAL_METRICS:
        df["BLEU"] = df.apply(lambda row: evaluate_bleu(row["tgt"], row["translated"]), axis=1)
        results["BLEU"] = df["BLEU"].mean()

    if "meteor" in EVAL_METRICS:
        df["METEOR"] = df.apply(lambda row: evaluate_meteor(row["tgt"], row["translated"]), axis=1)
        results["METEOR"] = df["METEOR"].mean()

    if "ter" in EVAL_METRICS:
        df["TER"] = df.apply(lambda row: evaluate_ter(row["tgt"], row["translated"]), axis=1)
        results["TER"] = df["TER"].mean()

    if "comet" in EVAL_METRICS and COMET_AVAILABLE:
        results["COMET"] = evaluate_comet(df)
    elif "comet" in EVAL_METRICS:
        results["COMET"] = "Not installed"

    print("=== Evaluation Results ===")
    for metric, score in results.items():
        print(f"{metric.upper()}: {score:.4f}" if isinstance(score, float) else f"{metric.upper()}: {score}")

if __name__ == "__main__":
    run_evaluation()
