
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "My payment failed yesterday."

# Let the tokenizer prepare the complete model input
encoded = tokenizer(text)

print("Original text:")
print(text)

print("\nTokens including special tokens:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"]))

print("\nInput IDs:")
print(encoded["input_ids"])

print("\nAttention mask:")
print(encoded["attention_mask"])

print("\nDecoded text:")
print(tokenizer.decode(encoded["input_ids"]))
