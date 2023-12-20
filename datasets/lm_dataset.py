import torch
from torch.utils.data import Dataset

from tokenizers import tokenizer, special_tokens

class LMDataset(Dataset):
    """
    Dataset for language model.

    Args:
        corpus (list[str]): The corpus on which the language model will be trained on.
        tokenizer (Tokenizer): Tokenizer for tokenizing the texts from the corpus.
        context_length (int): The context length of the langauge model.
    """
    def __init__(self, corpus: list[str], tokenizer: tokenizer.Tokenizer, context_length: int) -> None:
        """
        Initialize the dataset.

        Args:
            corpus (list[str]): The corpus on which the language model will be trained on.
            tokenizer (Tokenizer): Tokenizer for tokenizing the texts from the corpus.
            context_length (int): The context length of the langauge model.
        """
        self._tokenizer = tokenizer
        self._context_length = context_length

        tokenized_corpus: list[list[int]] = [tokenizer.encode(text) for text in corpus]
        tokenized_corpus = [text for text in tokenized_corpus if len(text) < context_length]

        self._corpus: list[list[int]] = []

        for encoded_text in tokenized_corpus:
            encoded_text_tensor = torch.zeros(self._context_length + 1, dtype=torch.long)

            encoded_text_tensor[1:len(encoded_text) + 1] = torch.tensor(encoded_text, dtype=torch.long)

            encoded_text_tensor[0] = self._tokenizer.stoi[special_tokens.SpecialTokens.SOS]
            encoded_text_tensor[self._context_length] = self._tokenizer.stoi[special_tokens.SpecialTokens.EOS]

            pad_tokens = [self._tokenizer.stoi[special_tokens.SpecialTokens.PAD] for _ in range(self._context_length - len(encoded_text) - 1)]
            encoded_text_tensor[len(encoded_text) + 1:self._context_length] = torch.tensor(pad_tokens, dtype=torch.long)

            self._corpus.append(encoded_text_tensor)

    def __len__(self)  -> int:
        """
        Returns the length of the dataset.
        """
        return len(self._corpus)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        """
        Returns the example at index idx. The token sequence will get start, end and padding tokens
        so that the sequence have the length context_length.

        Args:
            idx (int): The index of the example to be retrieved.
        """
        encoded_text: list[int] = self._corpus[idx]

        x = encoded_text[:self._context_length]
        y = encoded_text[1:self._context_length + 1]

        return x, y