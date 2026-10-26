
import pandas as pd
import matplotlib.pyplot as plt

BASE_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/"
    "task-specific-datasets/master/banking_data"
)

train_df = pd.read_csv(f"{BASE_URL}/train.csv")

# Count examples for each intent
category_counts = train_df["category"].value_counts()

print("Number of categories:", len(category_counts))
print("Minimum examples in a category:", category_counts.min())
print("Maximum examples in a category:", category_counts.max())
print("Average examples per category:", round(category_counts.mean(), 2))

print("\nFive categories with the fewest examples:")
print(category_counts.sort_values().head())

print("\nFive categories with the most examples:")
print(category_counts.sort_values(ascending=False).head())

# Plot the complete category distribution
plt.figure(figsize=(16, 6))

category_counts.sort_values().plot(kind="bar")

plt.title("Banking77 Training Data: Examples per Category")
plt.xlabel("Intent Category")
plt.ylabel("Number of Examples")
plt.xticks(rotation=90, fontsize=7)
plt.tight_layout()

plt.show()
