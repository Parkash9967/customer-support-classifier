
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from torch.optim import AdamW
from sklearn.metrics import accuracy_score


MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 77
MAX_LENGTH = 64
BATCH_SIZE = 8

BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data"
)


class BankingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item



def main():
    print("Loading dataset...")

    df = pd.read_csv(f"{BASE_URL}/train.csv")
    categories = sorted(df["category"].unique())
    label2id = {category: idx for idx, category in enumerate(categories)}
    df["label"] = df["category"].map(label2id)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = BankingDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )
    val_dataset = BankingDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    device = torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Train for one epoch initially
    epochs = 1
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )

            loss = loss_fn(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation: no gradient calculations or weight updates
        model.eval()
        val_predictions = []
        val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    key: value.to(device)
                    for key, value in batch.items()
                }

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                )

                predictions = outputs.logits.argmax(dim=1)

                val_predictions.extend(predictions.cpu().tolist())
                val_labels.extend(batch["labels"].cpu().tolist())

        val_accuracy = accuracy_score(val_labels, val_predictions)

        print(f"\nEpoch {epoch + 1} complete")
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Elapsed time: {(time.time() - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()

    print("Loading dataset...")

    df = pd.read_csv(f"{BASE_URL}/train.csv")
    categories = sorted(df["category"].unique())
    label2id = {category: idx for idx, category in enumerate(categories)}
    df["label"] = df["category"].map(label2id)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = BankingDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )
    val_dataset = BankingDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Number of classes:", len(categories))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # Inspect one batch
    batch = next(iter(train_loader))
    print("\nOne training batch:")
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

    # Load the classifier, but do not train yet
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )

    print("\nModel loaded successfully.")
    print("Parameters:", f"{sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
