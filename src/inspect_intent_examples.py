
from pathlib import Path

import pandas as pd

# Original Banking77 training dataset
DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "PolyAI-LDN/task-specific-datasets/master/"
    "banking_data/train.csv"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download and cache the dataset locally
TRAIN_FILE = DATA_DIR / "train.csv"

if TRAIN_FILE.exists():
    print(f"Loading cached dataset: {TRAIN_FILE}")
    df = pd.read_csv(TRAIN_FILE)
else:
    print("Downloading Banking77 training dataset from GitHub...")
    df = pd.read_csv(DATASET_URL)
    df.to_csv(TRAIN_FILE, index=False)
    print(f"Dataset saved to: {TRAIN_FILE}")

print(f"\nTotal training examples: {len(df)}")
print("Columns:", df.columns.tolist())

categories_to_inspect = [
    "lost_or_stolen_card",
    "compromised_card",
]

for category in categories_to_inspect:
    examples = df.loc[
        df["category"] == category,
        "text"
    ]

    print("\n" + "=" * 75)
    print(f"CATEGORY: {category} | EXAMPLES: {len(examples)}")
    print("=" * 75)

    for index, message in enumerate(examples, start=1):
        print(f"{index}. {message}")
