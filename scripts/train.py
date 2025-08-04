import os
import argparse
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from tqdm import tqdm

from config import (
    MODEL_NAME, DATA_DIR, TEST_DATASET, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    SOURCE_LANG, TARGET_LANG
)

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

def preprocess_function(examples, tokenizer):
    inputs = examples["src"]
    targets = examples["tgt"]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def evaluate_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)

def evaluate_meteor(reference, hypothesis):
    return meteor_score([reference], hypothesis)

def evaluate_comet(srcs, hypos, refs):
    if not COMET_AVAILABLE:
        return ["N/A"] * len(srcs)
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    samples = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hypos, refs)]
    scores = model.predict(samples, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
    return scores.scores

def translate(texts, tokenizer, model, src_lang, tgt_lang, batch_size=8):
    results = []
    tokenizer.src_lang = src_lang
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating {src_lang} → {tgt_lang}"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
    return results

def save_results(srcs, refs, hypos, bleu_scores, meteor_scores, comet_scores, output_dir):
    df = pd.DataFrame({
        "src": srcs,
        "tgt": refs,
        "translation": hypos,
        "BLEU": bleu_scores,
        "METEOR": meteor_scores,
        "COMET": comet_scores
    })
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(out_path, index=False)
    print(f"[✓] Saved detailed results to {out_path}")


def main(args):
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    train_dataset_path = os.path.join(DATA_DIR, args.dataset)
    raw_dataset = load_dataset("csv", data_files=train_dataset_path, split="train")

    tokenized_dataset = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # Replace -100 with tokenizer.pad_token_id in labels for decoding
        labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]

        preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)

        bleu = sum([evaluate_bleu(ref, hyp) for ref, hyp in zip(labels_text, preds_text)]) / len(preds_text)
        meteor = sum([evaluate_meteor(ref, hyp) for ref, hyp in zip(labels_text, preds_text)]) / len(preds_text)
        comet_scores = evaluate_comet(labels_text, preds_text, labels_text)
        comet_score = "N/A" if not COMET_AVAILABLE else sum(comet_scores) / len(comet_scores)

        print(f"[Epoch Eval] BLEU={bleu:.4f}, METEOR={meteor:.4f}, COMET={comet_score if isinstance(comet_score, str) else f'{comet_score:.4f}'}")

        return {
            "bleu": bleu,
            "meteor": meteor,
            "comet": comet_score if isinstance(comet_score, float) else 0.0,
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=EPOCHS,
        predict_with_generate=True,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        no_cuda=not torch.cuda.is_available()
    )

    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )


    print("[*] Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[✓] Model fine-tuned and saved to {args.output_dir}")

    # Load test data
    test_path = os.path.join(DATA_DIR, TEST_DATASET)
    test_df = pd.read_csv(test_path)
    src_texts = test_df["src"].tolist()
    tgt_texts = test_df["tgt"].tolist()

    print("[*] Starting translation on test data...")
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    translations = translate(src_texts, tokenizer, model, SOURCE_LANG, TARGET_LANG, batch_size=BATCH_SIZE)

    print("[*] Evaluating translations...")
    bleu_scores = [evaluate_bleu(ref, hyp) for ref, hyp in zip(tgt_texts, translations)]
    meteor_scores = [evaluate_meteor(ref, hyp) for ref, hyp in zip(tgt_texts, translations)]
    comet_scores = evaluate_comet(src_texts, translations, tgt_texts)

    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_comet = "N/A" if not COMET_AVAILABLE else sum(comet_scores) / len(comet_scores)

    print("\n=== Evaluation Results ===")
    print(f"Average BLEU:   {avg_bleu:.4f}")
    print(f"Average METEOR: {avg_meteor:.4f}")
    print(f"Average COMET:  {avg_comet if isinstance(avg_comet, str) else f'{avg_comet:.4f}'}")

    save_results(src_texts, tgt_texts, translations, bleu_scores, meteor_scores, comet_scores, output_dir=args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 & evaluate")
    parser.add_argument("--dataset", type=str, default="combined.csv", help="Training CSV in data/")
    parser.add_argument("--output_dir", type=str, default="model_finetuned", help="Directory for model + results")
    args = parser.parse_args()

    main(args)
