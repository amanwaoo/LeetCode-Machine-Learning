import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)

        # Define the architecture here
        self.ln1 = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.ln2 = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        out_images = self.dropout(self.relu(self.ln1(images)))
        out_images = self.sigmoid(self.ln2(out_images))
        return torch.round(out_images, decimals=4)
        # Return the model's prediction to 4 decimal places
