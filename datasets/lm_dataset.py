import torch
from torch.utils.data import Dataset

from unidecode import unidecode

from tokenizers import tokenizer

# token count
# special tokens
# corpus
# epochs

class LMDataset(Dataset):
    def __init__(self, tokenizer: tokenizer.Tokenizer, text: str, context_length: int):
        self._context_length = context_length

        unicode_text = unidecode(text)
        tokenized_text: list[int] = tokenizer.encode(unicode_text)
        self._data = torch.tensor(tokenized_text, dtype=torch.long)

        assert len(self._data) - context_length >= 0, 'Length of data is shorter than context length'

    def __len__(self):
        return len(self._data) - self._context_length

    def __getitem__(self, idx: int):
        x = [self._data[idx:idx + self._context_length]]
        y = [self._data[idx + 1:idx + self._context_length + 1]]
        return x, y