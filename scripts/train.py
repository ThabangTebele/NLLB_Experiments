import os
import argparse
import pandas as pd
from transformers import GenerationConfig
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
import traceback

from config import (
    MODEL_NAME, DATA_DIR, COMBINED_FILE, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    SOURCE_LANG, TARGET_LANG, DEVICE
)

# Try to import COMET for additional evaluation metric
try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

def preprocess_function(examples, tokenizer):
    """
    Tokenizes input and target texts for the model.
    """
    try:
        inputs = examples["src"]
        targets = examples["tgt"]
        model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    except Exception as e:
        print(f" Error in preprocess_function: {e}")
        traceback.print_exc()
        return {}

def evaluate_bleu(reference, hypothesis):
    """
    Computes BLEU score for a single reference-hypothesis pair.
    """
    try:
        smoothie = SmoothingFunction().method4
        return sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)
    except Exception as e:
        print(f" BLEU evaluation error: {e}")
        traceback.print_exc()
        return 0.0

def evaluate_meteor(reference, hypothesis):
    """
    Computes METEOR score for a single reference-hypothesis pair.
    """
    try:
        return meteor_score([reference.split()], hypothesis.split())

    except Exception as e:
        print(f" METEOR evaluation error: {e}")
        traceback.print_exc()
        return 0.0

def evaluate_comet(srcs, hypos, refs):
    """
    Computes COMET scores for a batch of source, hypothesis, and reference sentences.
    """
    if not COMET_AVAILABLE:
        return ["N/A"] * len(srcs)
    try:
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
        samples = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hypos, refs)]
        scores = model.predict(samples, batch_size=8, gpus=1 if torch.cuda.is_available() else 0)
        return scores.scores
    except Exception as e:
        print(f" COMET evaluation error: {e}")
        traceback.print_exc()
        return ["N/A"] * len(srcs)

def translate(texts, tokenizer, model, src_lang, tgt_lang,generation_config, batch_size=8):
    """
    Translates a list of texts from src_lang to tgt_lang using the provided model and tokenizer.
    """
    results = []

    # Set the target language
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Translating {src_lang} → {tgt_lang}"):
        batch = texts[i:i+batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                generation_config=generation_config
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(decoded)
        except Exception as e:
            print(f" Error during translation batch {i}-{i+batch_size}: {e}")
            traceback.print_exc()
            results.extend(["[ERROR]"] * len(batch))
    return results

def save_results(srcs, refs, hypos, bleu_scores, meteor_scores, comet_scores, output_dir):
    """
    Saves translation results and evaluation metrics to a CSV file in output_dir.
    """
    try:
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
        print(f" Saved detailed results to {out_path}")
    except Exception as e:
        print(f" Error saving results: {e}")
        traceback.print_exc()

def main(args):
    """
    Main training and evaluation workflow:
    - Loads data
    - Tokenizes and splits dataset
    - Fine-tunes the model
    - Evaluates on test data
    - Saves results
    """
    import torch
    # Load tokenizer and model
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        tokenizer.src_lang = "eng_Latn"
        tokenizer.tgt_lang = "nso_Latn"

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    except Exception as e:
        print(f" Error loading model/tokenizer: {e}")
        traceback.print_exc()
        return

    # Load and optionally limit the dataset
    try:
        train_dataset_path = os.path.join(COMBINED_FILE, args.dataset)
        print(f"Loading dataset from: {train_dataset_path}")

        raw_dataset = load_dataset(
        "csv",
        data_files={"train": "data/combined.csv"},
        split="train"
        )

        print("Number of training examples:", len(raw_dataset))
        print("Sample data:", raw_dataset[0] if len(raw_dataset) > 0 else "No data found")


        if args.limit:
            limit_val = min(args.limit, len(raw_dataset))
            raw_dataset = raw_dataset.select(range(limit_val))
            print(f" Dataset truncated to {limit_val} samples for quick testing.")
    except Exception as e:
        print(f" Error loading dataset: {e}")
        traceback.print_exc()
        return
    
    generation_config = GenerationConfig.from_pretrained(args.output_dir)


    # Tokenize and split dataset
    try:
        tokenized_dataset = raw_dataset.map(
            lambda x: preprocess_function(x, tokenizer), 
            batched=True
            #remove_columns=raw_dataset.column_names
        )
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    except Exception as e:
        print(f" Error during dataset tokenization/splitting: {e}")
        traceback.print_exc()
        return

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Metric computation for validation during training
    def compute_metrics(eval_preds):
        try:
            preds, labels = eval_preds
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
        except Exception as e:
            print(f" Error in compute_metrics: {e}")
            traceback.print_exc()
            return {"bleu": 0.0, "meteor": 0.0, "comet": 0.0}

    # Set up training arguments for HuggingFace Trainer
    try:
        training_args = Seq2SeqTrainingArguments(
            output_dir=args.output_dir,
            eval_strategy="epoch",
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
            use_cpu=not torch.cuda.is_available()
        )
    except Exception as e:
        print(f" Error setting up training arguments: {e}")
        traceback.print_exc()
        return

    # Initialize Trainer and start training
    try:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print(" Starting training...")
        trainer.train()
        trainer.save_model(args.output_dir)

        # Save GenerationConfig explicitly
        gen_config = GenerationConfig.from_model_config(model.config)



        #(Optional) Customize generation parameters
        gen_config.max_length = MAX_LENGTH
        gen_config.num_beams = 4  # or whatever you're using
        gen_config.early_stopping = True
        gen_config.decoder_start_token_id = model.config.decoder_start_token_id

        gen_config.save_pretrained(args.output_dir)
        print(" Saved generation config to:", args.output_dir)


        print(f" Model fine-tuned and saved to {args.output_dir}")
    except Exception as e:
        print(f" Error during training: {e}")
        traceback.print_exc()
        return

    # Load test data for final evaluation
    try:
        test_path = COMBINED_FILE
        test_df = pd.read_csv(test_path)

        test_df.columns = test_df.columns.str.strip()  # Clean column names just in case
        print("Available columns in test_df:", test_df.columns.tolist())  # Debug output

        src_texts = test_df["src"].tolist()
        tgt_texts = test_df["tgt"].tolist()
    except Exception as e:
        print(f" Error loading test data: {e}")
        traceback.print_exc()
        return

    print(" Starting translation on test data...")
    # Translate test data
    try:
        model.eval()
        model.to(DEVICE)
        translations = translate(src_texts, tokenizer, model, SOURCE_LANG, TARGET_LANG,generation_config=generation_config, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f" Error during translation: {e}")
        traceback.print_exc()
        return

    print(" Evaluating translations...")
    # Evaluate and save results
    try:
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
    except Exception as e:
        print(f" Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command-line arguments for dataset, output directory, and optional limit
    parser = argparse.ArgumentParser(description="Fine-tune NLLB-200 & evaluate")
    parser.add_argument("--dataset", type=str, default="combined.csv", help="Training CSV in data/")
    parser.add_argument("--output_dir", type=str, default="model_finetuned", help="Directory for model + results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of training samples for quick tests")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f" Unexpected error: {e}")
        traceback.print_exc()
