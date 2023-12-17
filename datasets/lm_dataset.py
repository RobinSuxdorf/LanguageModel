import torch
from torch.utils.data import Dataset

from tokenizers import tokenizer, special_tokens

# stop generation after SOS
# move logic from __getiem__ to __init__ or save it directly in json
#    can assert that all entries have length _context_length + 1
# context_length as property of language model class
# doc strings
# printing of PAD tokens -> remove from sampling
# estimated time

class LMDataset(Dataset):
    """
    Dataset for language model.

    Args:
        corpus (list[str]): Corpus containing texts.
        tokenizer (Tokenizer): Tokenizer for tokenizing the texts from the corpus
        context_length (int): The context length of the langauge model.
    """
    def __init__(self, corpus: list[str], tokenizer: tokenizer.Tokenizer, context_length: int):
        self._tokenizer = tokenizer
        self._context_length = context_length

        tokenized_corpus: list[list[int]] = [tokenizer.encode(text) for text in corpus]
        self._corpus: list[list[int]] = [text for text in tokenized_corpus if len(text) < context_length]

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