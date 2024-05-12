import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocab = set()
        for sentence in positive:
            for words in sentence.split():
                vocab.add(words)
        for sentence in negative:
            for words in sentence.split():
                vocab.add(words)

        list_sort = sorted(list(vocab))
        list_to_int = {}
        for i in range(len(list_sort)):
            list_to_int[list_sort[i]] = i + 1
        
        list_tensor = []
        
        for sentence in positive:
            curr_list_tensor = []
            for words in sentence.split():
                curr_list_tensor.append(list_to_int[words])
            list_tensor.append(torch.tensor(curr_list_tensor))

        for sentence in negative:
            curr_list_tensor = []
            for words in sentence.split():
                curr_list_tensor.append(list_to_int[words])
            list_tensor.append(torch.tensor(curr_list_tensor))

        return nn.utils.rnn.pad_sequence(list_tensor, batch_first=True)
