
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "banking77-bert"
DATA_FILE = PROJECT_ROOT / "data" / "train.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 64
BATCH_SIZE = 16


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    ).to(DEVICE)
    model.eval()

    label_names = np.load(
        MODEL_DIR / "label_names.npy", allow_pickle=True
    ).tolist()

    df = pd.read_csv(DATA_FILE).dropna(
        subset=["text", "category"]
    ).copy()

    _, val_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["category"],
    )
    val_df = val_df.reset_index(drop=True)

    records = []

    for start in range(0, len(val_df), BATCH_SIZE):
        batch = val_df.iloc[start:start + BATCH_SIZE]
        texts = batch["text"].astype(str).tolist()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.inference_mode():
            probabilities = torch.softmax(
                model(**inputs).logits, dim=1
            )

        top_probs, top_indices = torch.topk(
            probabilities, k=3, dim=1
        )

        for i, (_, row) in enumerate(batch.iterrows()):
            predicted = label_names[top_indices[i, 0].item()]

            records.append({
                "text": row["text"],
                "actual": row["category"],
                "predicted": predicted,
                "confidence": top_probs[i, 0].item(),
                "top2_intent": label_names[top_indices[i, 1].item()],
                "top2_confidence": top_probs[i, 1].item(),
                "top3_intent": label_names[top_indices[i, 2].item()],
                "top3_confidence": top_probs[i, 2].item(),
                "correct": predicted == row["category"],
            })

        print(f"Evaluated {min(start + len(batch), len(val_df))}/{len(val_df)}")

    results = pd.DataFrame(records)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = REPORTS_DIR / "validation_confidence_analysis.csv"
    results.to_csv(output_file, index=False)

    print("\nCONFIDENCE THRESHOLD ANALYSIS")
    print("-" * 75)
    print(
        f"{'Threshold':>10} {'Coverage':>10} "
        f"{'Accepted':>10} {'Accuracy':>10} {'Errors':>10}"
    )

    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        accepted = results[results["confidence"] >= threshold]
        coverage = len(accepted) / len(results)
        accuracy = (
            accuracy_score(accepted["actual"], accepted["predicted"])
            if len(accepted) else 0
        )
        errors = int((~accepted["correct"]).sum())

        print(
            f"{threshold:>9.0%} "
            f"{coverage:>9.1%} "
            f"{len(accepted):>10} "
            f"{accuracy:>9.1%} "
            f"{errors:>10}"
        )

    print(f"\nSaved detailed results to: {output_file}")
    
        # Inspect high-confidence mistakes
    print("\nHIGH-CONFIDENCE MISTAKES (confidence >= 0.80)")
    print("-" * 75)

    high_confidence_errors = results[
        (results["confidence"] >= 0.80)
        & (~results["correct"])
    ].sort_values("confidence", ascending=False)

    for _, row in high_confidence_errors.iterrows():
        print(f"\nMessage:    {row['text']}")
        print(f"Actual:     {row['actual']}")
        print(f"Predicted:  {row['predicted']}")
        print(f"Confidence: {row['confidence']:.2%}")
        print(
            f"Alternative: {row['top2_intent']} "
            f"({row['top2_confidence']:.2%})"
        )


if __name__ == "__main__":
    main()
