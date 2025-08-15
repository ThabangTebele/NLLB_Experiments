import os
import argparse
import pandas as pd
import numpy as np
import torch
import traceback
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    GenerationConfig
)
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from config import MODEL_NAME, COMBINED_FILE, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, DEVICE

CHECKPOINT_DIR = "models/nllb-finetuned"

# --- Metrics ---
def evaluate_bleu(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)

def evaluate_meteor(reference, hypothesis):
    return meteor_score([reference.split()], hypothesis.split())

# --- Preprocessing ---
def preprocess_function(examples, tokenizer):
    inputs = examples["src"]
    targets = examples["tgt"]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=MAX_LENGTH, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- Metrics computation ---
def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    bleu = np.mean([evaluate_bleu(r, p) for r, p in zip(labels_text, preds_text)])
    meteor = np.mean([evaluate_meteor(r, p) for r, p in zip(labels_text, preds_text)])
    return {"bleu": bleu, "meteor": meteor}

def main(args):
    # --- Load dataset ---
    raw_dataset = load_dataset("csv", data_files={"train": str(COMBINED_FILE)}, split="train")
    if args.limit:
        raw_dataset = raw_dataset.select(range(min(args.limit, len(raw_dataset))))

    # --- Detect source and target languages from dataset ---
    src_lang = raw_dataset[0]["src_lang"]
    tgt_lang = raw_dataset[0]["tgt_lang"]
    print(f"Detected languages -> Source: {src_lang}, Target: {tgt_lang}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=src_lang,
        tgt_lang=tgt_lang
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)

    tokenized_dataset = raw_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        eval_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        use_cpu=not torch.cuda.is_available()
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )

    # --- Resume from last checkpoint if available ---
    last_checkpoint = None
    if os.path.isdir(CHECKPOINT_DIR) and any(os.scandir(CHECKPOINT_DIR)):
        checkpoints = [d.path for d in os.scandir(CHECKPOINT_DIR) if d.is_dir()]
        if checkpoints:
            try:
                last_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
                print(f"Resuming training from checkpoint: {last_checkpoint}")
            except ValueError:
                print("Warning: Could not parse checkpoint numbers. Starting fresh.")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(args.output_dir)

    gen_config = GenerationConfig.from_model_config(model.config)
    gen_config.save_pretrained(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 with checkpoint support and auto lang detection")
    parser.add_argument("--output_dir", type=str, default="models/nllb-finetuned", help="Directory for saving model/checkpoints")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for number of training samples")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
