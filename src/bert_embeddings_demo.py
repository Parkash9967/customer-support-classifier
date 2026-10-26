
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Two sentences with different contexts
texts = [
    "My payment failed.",
    "My payment was successful."
]

# Convert both sentences into model inputs
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

model.eval()

with torch.no_grad():
    # Embeddings before the Transformer encoder
    input_embeddings = model.embeddings(
        inputs["input_ids"]
    )

    # Output after all BERT Transformer layers
    outputs = model(**inputs)

    contextual_embeddings = outputs.last_hidden_state

print("Input IDs shape:")
print(inputs["input_ids"].shape)

print("\nInput embeddings shape:")
print(input_embeddings.shape)

print("\nContextual embeddings shape:")
print(contextual_embeddings.shape)

# Find the token ID for "payment"
payment_id = tokenizer.convert_tokens_to_ids("payment")

for i, text in enumerate(texts):
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][i]
    )

    payment_position = tokens.index("payment")

    before = input_embeddings[i, payment_position]
    after = contextual_embeddings[i, payment_position]

    difference = torch.norm(after - before).item()

    print(f"\nSentence: {text}")
    print("Tokens:", tokens)
    print("Payment token ID:", payment_id)
    print("Payment position:", payment_position)
    print("Input embedding (first 5 values):")
    print(before[:5])
    print("Contextual representation (first 5 values):")
    print(after[:5])
    print("L2 difference:", round(difference, 4))
