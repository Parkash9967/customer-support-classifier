
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ERROR_FILE = PROJECT_ROOT / "reports" / "validation_misclassifications.csv"

df = pd.read_csv(ERROR_FILE)

categories_to_inspect = [
    "top_up_failed",
    "top_up_reverted",
    "failed_transfer",
    "declined_transfer",
    "virtual_card_not_working",
]

# Show mistakes where either the actual or predicted label
# belongs to one of our selected categories.
errors = df[
    df["category"].isin(categories_to_inspect)
    | df["predicted_category"].isin(categories_to_inspect)
]

print(f"Total validation mistakes: {len(df)}")
print(f"Relevant mistakes to inspect: {len(errors)}")

for _, row in errors.iterrows():
    print("\n" + "-" * 80)
    print(f"Message:   {row['text']}")
    print(f"Actual:    {row['category']}")
    print(f"Predicted: {row['predicted_category']}")
