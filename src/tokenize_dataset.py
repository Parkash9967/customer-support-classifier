
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data"
)

# Load the official training dataset
df = pd.read_csv(f"{BASE_URL}/train.csv")

# Create consistent numeric labels
categories = sorted(df["category"].unique())
label2id = {category: idx for idx, category in enumerate(categories)}
df["label"] = df["category"].map(label2id)

# Create training and validation sets
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

# Load the pretrained BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Select a small batch of real customer messages
texts = train_df["text"].head(4).tolist()
labels = train_df["label"].head(4).tolist()

# Convert text into model inputs
encoded = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
)

print("Original messages:")
for text in texts:
    print("-", text)

print("\nNumeric labels:")
print(labels)

print("\nTokenized tensors:")
print("input_ids shape:", encoded["input_ids"].shape)
print("attention_mask shape:", encoded["attention_mask"].shape)

print("\nFirst message tokens:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))

print("\nFirst message input IDs:")
print(encoded["input_ids"][0].tolist())

print("\nFirst message attention mask:")
print(encoded["attention_mask"][0].tolist())
