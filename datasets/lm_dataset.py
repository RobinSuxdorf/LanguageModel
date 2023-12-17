import torch
from torch.utils.data import Dataset

from tokenizers import tokenizer, special_tokens

# stop generation after SOS
# notebook for creating dataset -> token count
# move logic from __getiem__ to __init__ or save it directly in json
#    can assert that all entries have length _context_length + 1
# context_length as property of language model class
# doc strings
# printing of PAD tokens
# estimated time

class LMDataset(Dataset):
    def __init__(self, tokenizer: tokenizer, corpus: list[list[int]], context_length: int):
        self._tokenizer = tokenizer
        self._context_length = context_length
        self._corpus = corpus

    def __len__(self):
        return len(self._corpus)

    def __getitem__(self, idx: int):
        encoded_text: list[int] = self._corpus[idx]
        
        for _ in range(self._context_length - len(encoded_text) - 1):
            encoded_text.append(self._tokenizer.stoi[special_tokens.SpecialTokens.PAD])

        encoded_text.insert(0, self._tokenizer.stoi[special_tokens.SpecialTokens.SOS])
        encoded_text.append(self._tokenizer.stoi[special_tokens.SpecialTokens.EOS])

        encoded_text = torch.tensor(encoded_text, dtype=torch.long)

        x = encoded_text[:self._context_length]
        y = encoded_text[1:self._context_length + 1]

        return x, y