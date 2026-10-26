
import torch
import torch.nn as nn

# Token IDs from our previous BERT tokenizer experiment
input_ids = torch.tensor([
    [101, 2026, 7909, 3478, 7483, 1012, 102]
])

# Create a small embedding layer:
# 30522 possible token IDs, each mapped to an 8-dimensional vector
embedding_layer = nn.Embedding(
    num_embeddings=30522,
    embedding_dim=8
)

# Convert token IDs into embedding vectors
embeddings = embedding_layer(input_ids)

print("Input IDs shape:")
print(input_ids.shape)

print("\nEmbeddings shape:")
print(embeddings.shape)

print("\nEmbedding vector for the first token ([CLS]):")
print(embeddings[0, 0])

print("\nEmbedding vector for the second token ('my'):")
print(embeddings[0, 1])
