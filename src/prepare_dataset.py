
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data"
)

# Load the official datasets
train_df = pd.read_csv(f"{BASE_URL}/train.csv")
test_df = pd.read_csv(f"{BASE_URL}/test.csv")

# Create a consistent mapping from category names to numeric IDs
categories = sorted(train_df["category"].unique())

label2id = {
    category: idx
    for idx, category in enumerate(categories)
}

id2label = {
    idx: category
    for category, idx in label2id.items()
}

# Convert text labels to integer labels
train_df["label"] = train_df["category"].map(label2id)
test_df["label"] = test_df["category"].map(label2id)

# Split only the official training data
train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=42,
    stratify=train_df["label"],
)

print("Training examples:", len(train_split))
print("Validation examples:", len(val_split))
print("Official test examples:", len(test_df))
print("Number of classes:", len(categories))

print("\nTraining label sample:")
print(train_split[["text", "category", "label"]].head())

print("\nMissing labels:")
print("Training:", train_split["label"].isna().sum())
print("Validation:", val_split["label"].isna().sum())
print("Test:", test_df["label"].isna().sum())
