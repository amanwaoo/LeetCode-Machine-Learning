import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        # You must start by generating batch_size different random indices in the appropriate range
        # using a single call to torch.randint()
        torch.manual_seed(0)
        tokenizer = raw_dataset.split()
        random_index = torch.randint(high=len(tokenizer)-context_length, size=(batch_size,)).tolist()
        X, Y = [], []
        for i in random_index:
            X.append(tokenizer[i:i+context_length])
            Y.append(tokenizer[i+1:i+1+context_length])
        return X,Y
