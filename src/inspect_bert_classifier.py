
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 77

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
)

texts = [
    "How long do transfers take to show up in my account?",
    "My virtual card is not working.",
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt",
)

model.eval()

with torch.no_grad():
    outputs = model(**inputs)

print("Input IDs shape:", inputs["input_ids"].shape)
print("Logits shape:", outputs.logits.shape)
print("Predicted class IDs:", outputs.logits.argmax(dim=1).tolist())
print("Number of output classes:", model.config.num_labels)
