
import pandas as pd

BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data"
)

# Read the official CSV files directly
train_df = pd.read_csv(f"{BASE_URL}/train.csv")
test_df = pd.read_csv(f"{BASE_URL}/test.csv")

print("Training dataset shape:", train_df.shape)
print("Test dataset shape:", test_df.shape)

print("\nTraining columns:")
print(train_df.columns.tolist())

print("\nFirst 5 training examples:")
print(train_df.head())

print("\nTraining data types:")
print(train_df.dtypes)

# Inspect the target labels
print("\nNumber of unique intent categories:")
print(train_df["category"].nunique())

print("\nMost common 10 categories:")
print(train_df["category"].value_counts().head(10))

print("\nMissing values:")
print(train_df.isnull().sum())
