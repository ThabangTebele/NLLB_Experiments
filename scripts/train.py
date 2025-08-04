import os
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)
from config import MODEL_NAME, DATA_DIR, PROCESSED_DIR, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE, SOURCE_LANG, TARGET_LANG

def preprocess_function(examples, tokenizer):
    inputs = examples["src"]
    targets = examples["tgt"]

    model_inputs = tokenizer(
        inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length"
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=MAX_LENGTH, truncation=True, padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Load combined dataset CSV
    dataset_path = os.path.join(DATA_DIR, args.dataset)
    raw_dataset = load_dataset("csv", data_files=dataset_path, split="train")

    # Tokenize dataset
    tokenized_dataset = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer), 
        batched=True,
        remove_columns=raw_dataset.column_names
    )

    # Split train/val (90% train, 10% val)
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[✓] Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 with your datasets")
    parser.add_argument("--dataset", type=str, default="combined.csv", help="CSV dataset filename in data/")
    parser.add_argument("--output_dir", type=str, default="model_finetuned", help="Output directory for model")
    args = parser.parse_args()

    main(args)
