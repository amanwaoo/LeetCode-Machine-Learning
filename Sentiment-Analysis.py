import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.embedding_layer = nn.Embedding(vocabulary_size,16)
        self.linear = nn.Linear(16,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer
        embeddings = self.embedding_layer(x)
        averaged_embeddings = torch.mean(embeddings,1)
        linear_embeddings = self.linear(averaged_embeddings)
        # Return a B, 1 tensor and round to 4 decimal places
        projected_embeddings = self.sigmoid(linear_embeddings)

        return torch.round(projected_embeddings, decimals=4)
