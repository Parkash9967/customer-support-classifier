
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "banking77-bert"
DATA_FILE = PROJECT_ROOT / "data" / "train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64
BATCH_SIZE = 16


def load_model():
    print(f"Loading model from: {MODEL_DIR}")
    print(f"Evaluation device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )
    model.to(DEVICE)
    model.eval()

    # Prefer the label names saved with the project.
    label_file = MODEL_DIR / "label_names.npy"

    if label_file.exists():
        label_names = np.load(label_file, allow_pickle=True).tolist()
    else:
        id2label = model.config.id2label
        label_names = [
            id2label[i] if i in id2label else id2label[str(i)]
            for i in range(model.config.num_labels)
        ]

    return tokenizer, model, label_names


def load_validation_data():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna(subset=["text", "category"]).copy()

    _, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["category"],
    )

    val_df = val_df.reset_index(drop=True)
    print(f"Validation set loaded: {len(val_df)} examples")

    return val_df


def evaluate_model(val_df, tokenizer, model, label_names):
    all_predictions = []
    start_time = time.time()

    texts = val_df["text"].astype(str).tolist()

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)

        all_predictions.extend(predictions.cpu().tolist())

        completed = min(start + len(batch_texts), len(texts))
        if completed % 500 < BATCH_SIZE or completed == len(texts):
            print(f"Evaluated {completed}/{len(texts)} examples")

    predicted_labels = [label_names[i] for i in all_predictions]
    actual_labels = val_df["category"].astype(str).tolist()

    accuracy = accuracy_score(actual_labels, predicted_labels)
    macro_f1 = f1_score(
        actual_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 65)
    print("VALIDATION RESULTS")
    print("=" * 65)
    print(f"Validation examples: {len(val_df)}")
    print(f"Accuracy:            {accuracy:.4f} ({accuracy:.2%})")
    print(f"Macro F1:            {macro_f1:.4f}")
    print(f"Evaluation time:     {elapsed:.1f} seconds")
    print(f"Device:              {DEVICE}")

    results_df = val_df.copy()
    results_df["predicted_category"] = predicted_labels
    results_df["correct"] = (
        results_df["category"].astype(str)
        == results_df["predicted_category"]
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save every incorrect validation prediction for error analysis.
    mistakes = results_df[~results_df["correct"]].copy()
    mistakes.to_csv(
        REPORTS_DIR / "validation_misclassifications.csv",
        index=False,
    )

    # Save precision, recall, F1 and support for all 77 categories.
    report = classification_report(
        actual_labels,
        predicted_labels,
        labels=label_names,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        REPORTS_DIR / "validation_classification_report.csv",
        index=True,
    )

    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT — LOWEST F1 CATEGORIES")
    print("=" * 65)

    per_class = report_df.loc[label_names].copy()
    per_class = per_class.sort_values("f1-score")

    print(
        per_class[["precision", "recall", "f1-score", "support"]]
        .head(15)
        .to_string(float_format=lambda value: f"{value:.3f}")
    )

    print("\n" + "=" * 65)
    print("ERROR ANALYSIS FILES")
    print("=" * 65)
    print(f"Misclassified examples: {len(mistakes)}")
    print(f"Saved: {REPORTS_DIR / 'validation_misclassifications.csv'}")
    print(f"Saved: {REPORTS_DIR / 'validation_classification_report.csv'}")

    return results_df


def main():
    tokenizer, model, label_names = load_model()
    val_df = load_validation_data()

    print("\nStarting evaluation...")
    evaluate_model(val_df, tokenizer, model, label_names)


if __name__ == "__main__":
    main()
