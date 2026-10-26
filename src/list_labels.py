
from pathlib import Path

from transformers import AutoModelForSequenceClassification

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "banking77-bert"
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR)
)

for label_id, label in sorted(
    model.config.id2label.items(),
    key=lambda item: int(item[0])
):
    print(f"{label_id}: {label}")
