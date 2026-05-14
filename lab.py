"""
Module 7 Week A — Applied Lab: Fine-Tune DistilBERT for App-Review Sentiment.

Implement the TODO functions to build a complete fine-tuning pipeline.

Default run: `python lab.py` reads `data/app_reviews_train.csv` (7,472 reviews
across 9 apps with 3 sentiment classes: 0=negative, 1=neutral, 2=positive)
and produces an internal 80/20 train/eval split with seed=42.

CI smoke run: workflow sets DATA_PATH=fixtures/tiny_app_reviews.csv (60 rows).

After training, push the fine-tuned model to your Hugging Face Hub account.
The model directory is local-only (gitignored).
"""

import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, disable_caching
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers import set_seed
from sklearn.metrics import precision_score, recall_score

# Patch the Hasher.hash method to avoid dill pickling with Python 3.14
from datasets import fingerprint as fp_module
import hashlib

# Store the original hash method
_original_hash = fp_module.Hasher.hash

@classmethod
def _patched_hash(cls, value):
    """Hash function that avoids using dill to pickle objects"""
    try:
        # For simple types, just convert to string
        hash_str = str(value)
    except Exception:
        # For complex types, use a generic type-based hash
        hash_str = f"{type(value).__name__}_{id(value)}"
    return cls.hash_bytes(hash_str.encode())

fp_module.Hasher.hash = _patched_hash

# Disable dataset caching to avoid additional pickling issues
disable_caching()

# Label mappings for the 3-class sentiment classification task
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

def get_data_path() -> str:
    """
    Return DATA_PATH env var if set (CI uses a smoke CSV); otherwise return
    the default path to the curated app-review training CSV.

    Provided helper. Do not modify.
    """
    return os.environ.get("DATA_PATH", "data/app_reviews_train.csv")


def prepare_dataset(DATA_PATH: str, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna().reset_index(drop=True)

    # Use from_dict to avoid arrow table issues
    dataset = Dataset.from_dict({
        "text": df["text"].tolist(),
        "label": df["label"].tolist()
    })

    return dataset.train_test_split(test_size=test_size, seed=seed)

def tokenize_dataset(ds_dict: DatasetDict, tokenizer, max_length: int = 128) -> DatasetDict:
    """
    Tokenize all splits in a DatasetDict.

    `tokenizer` is a loaded HuggingFace tokenizer (callable) — load it once
    in `main()` via `AutoTokenizer.from_pretrained(...)` and pass it in.
    Use truncation=True and max_length=max_length. Do not pad here — padding is
    applied dynamically by DataCollatorWithPadding at training time.

    Note: this signature differs from the drill (`tokenize_dataset(ds, name)`)
    by accepting the loaded tokenizer object so `main()` doesn't re-load it.
    """
    # TODO: define tokenize_fn(batch) calling the passed-in tokenizer with truncation + max_length
    # TODO: apply ds_dict.map(tokenize_fn, batched=True)
    # TODO: return the tokenized DatasetDict
    
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length
        )

    return ds_dict.map(
        tokenize_fn,
        batched=True
    )

def make_training_args(
    output_dir: str,
    lr: float = 5e-5,
    epochs: int = 2,
    batch_size: int = 8,
    seed: int = 42,
) -> TrainingArguments:
    """Return a TrainingArguments configured for fine-tuning."""
    # TODO: return a TrainingArguments configured with the passed arguments.
    # In addition to wiring the kwargs through, set:
    #   - eval_strategy="epoch"           (renamed from evaluation_strategy in transformers 4.41+)
    #   - save_strategy="epoch"
    #   - logging_steps=50
    # The course pins transformers>=4.41,<5.0 — use the new argument names.
   
    training_args=TrainingArguments(
         output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        seed=seed,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        push_to_hub=False,
    )
    training_args.eval_strategy="epoch"
    training_args.save_strategy="epoch"
    return training_args
   

def compute_metrics(eval_pred):
    """
    Convert (logits, labels) into {"accuracy": ..., "macro_f1": ...}.

    Use sklearn's accuracy_score and f1_score with average="macro".
    """
    # TODO: unpack eval_pred to logits, labels
    # TODO: argmax logits over axis 1
    # TODO: compute accuracy and macro-F1
    # TODO: return as a dict
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    
    return {"accuracy": acc, "macro_f1": f1}
    


def train_classifier(
    tokenized_ds: DatasetDict,
    model_name: str,
    training_args: TrainingArguments,
    tokenizer,
    num_labels: int = 3,
) -> Trainer:
    """
    Construct and train a Trainer.

    Returns the trained Trainer (trainer.model is the fine-tuned model). Pass
    id2label=ID2LABEL and label2id=LABEL2ID to the model so its config records
    the human-readable label names — Integration 7A reads them from
    `model.config.id2label` rather than hard-coding.
    """
    # TODO: load model with AutoModelForSequenceClassification.from_pretrained(
    #         model_name, num_labels=num_labels, id2label=ID2LABEL, label2id=LABEL2ID)
    # TODO: build data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # TODO: build Trainer with model, args, train/eval datasets, tokenizer, data_collator, compute_metrics
    # TODO: call trainer.train()
    # TODO: return trainer
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        id2label=ID2LABEL, 
        label2id=LABEL2ID
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    return trainer


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classifier(trainer: Trainer, tokenized_test) -> dict:
    """
    Evaluate the trainer's model on the test split.

    Read label names from trainer.model.config.id2label (do not hard-code).

    Returns: {"accuracy": float, "macro_f1": float, "per_class_f1": {label_name: f1, ...}}
    """
    # TODO: predict on tokenized_test using trainer.predict
    # TODO: argmax predictions to class indices
    # TODO: compute accuracy and macro-F1
    # TODO: compute per-class F1 with f1_score(..., average=None)
    # TODO: build per_class_f1 dict using trainer.model.config.id2label for label names
    # TODO: return all three
    


    predictions_output = trainer.predict(tokenized_test)

    logits = predictions_output.predictions
    y_true = predictions_output.label_ids

    y_pred = np.argmax(logits, axis=-1)

    id2label = trainer.model.config.id2label

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    per_class_f1_scores = f1_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    per_class_precision_scores = precision_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    per_class_recall_scores = recall_score(
        y_true,
        y_pred,
        average=None,
        zero_division=0
    )

    per_class_f1 = {
        id2label[i]: float(per_class_f1_scores[i])
        for i in range(len(per_class_f1_scores))
    }

    per_class_precision = {
        id2label[i]: float(per_class_precision_scores[i])
        for i in range(len(per_class_precision_scores))
    }

    per_class_recall = {
        id2label[i]: float(per_class_recall_scores[i])
        for i in range(len(per_class_recall_scores))
    }

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
    }



def main() -> None:
    """Orchestrate the full pipeline."""
    data_path = get_data_path()
    output_dir = "model"
    model_name = "distilbert-base-uncased"
    set_seed(42)

    ds = prepare_dataset(data_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_dataset(ds, tokenizer)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    

    training_args = make_training_args(output_dir)
    trainer = train_classifier(tokenized, model_name, training_args, tokenizer, num_labels=3)

    # Save locally (model/ is gitignored)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    metrics = evaluate_classifier(trainer, tokenized["test"])
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV
    pred_logits = trainer.predict(tokenized["test"]).predictions
    pred_idx = np.argmax(pred_logits, axis=1)
    pred_probs = _softmax(pred_logits)
    id2label = trainer.model.config.id2label
    df_out = pd.DataFrame({
    "text": ds["test"]["text"],
    "label": [id2label[i] for i in ds["test"]["label"]],
    "predicted_label": [id2label[i] for i in pred_idx],
    "predicted_probability": [
        float(pred_probs[i, pred_idx[i]])
        for i in range(len(pred_idx))
    ],
    "prob_negative": pred_probs[:, 0],
    "prob_neutral": pred_probs[:, 1],
    "prob_positive": pred_probs[:, 2],
})
    df_out.to_csv("predictions.csv", index=False)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")

    # Confusion matrix (for the evaluation report)
    print("\nConfusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(
        [id2label[i] for i in ds["test"]["label"]],
        [id2label[i] for i in pred_idx],
        labels=list(id2label.values()),
    )
    print(pd.DataFrame(cm, index=list(id2label.values()), columns=list(id2label.values())).to_string())
    cm_df = pd.DataFrame(
    cm,
    index=list(id2label.values()),
    columns=list(id2label.values())
)

    cm_df.to_csv("confusion_matrix.csv")
    # Push to Hugging Face Hub.
    # Skipped in CI (DATA_PATH set); requires `huggingface-cli login` locally.
    if os.environ.get("DATA_PATH") is None:
        repo_id = "m7-app-review-sentiment"
        try:
            trainer.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)
            print(f"\nPushed to https://huggingface.co/<your-username>/{repo_id}")
        except Exception as e:
            print(f"\nHF Hub push failed: {e}")
            print("Run `huggingface-cli login` and try again.")


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last dimension."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)



if __name__ == "__main__":
    main()
