from dataclasses import dataclass

import pickle

import torch
import torch.nn as nn

from tokenizer import tokenizer
from language_model import model

@dataclass
class ModelArgs():
    context_length: int = 256
    embed_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    forward_expansion: int = 4
    dropout: float = 0.2

class LanguageModel():
    """
    Language model using encoder-only design.

    Args:
        tokenizer (Tokenizer): The tokenizer for encoding and decoding the text.
        device (str): The device where the calculations will be done, i.e. CPU or CUDA.
        args (ModelArgs): The model hyperparameters.
    """
    def __init__(
        self,
        tokenizer: tokenizer.Tokenizer,
        device: str,
        args: ModelArgs = ModelArgs()
    ):
        """
        Initialize the language model.

        Args:
            tokenizer (Tokenizer): The tokenizer for encoding and decoding the text.
            device (str): The device where the calculations will be done, i.e. CPU or CUDA.
            args (ModelArgs): The model hyperparameters.
        """
        self.tokenizer = tokenizer
        self.context_length = args.context_length
        self._device = device

        self.encoder = model.Encoder(
            tokenizer.vocab_size,
            args.embed_size,
            args.context_length,
            args.num_layers,
            args.num_heads,
            args.forward_expansion,
            args.dropout,
            device = device
        ).to(device)

    def predict(self, input_text: str, max_new_tokens: int) -> str:
        """
        Completes the input string based on that context.

        Args:
            input_text (str): The text which should be completed.
            max_new_tokens (int): Number of tokens that will be generated at most.

        Returns:
            str: The completed string.
        """

        context = torch.tensor(self.tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(self._device)

        for _ in range(max_new_tokens):
            idx_cond = context[:, -self.context_length:]

            logits, _ = self.encoder(idx_cond)

            logits = logits[:, -1, :]

            probs = nn.functional.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            context = torch.cat((context, idx_next), dim=1)

        output = self.tokenizer.decode(context[0].tolist())
        return output

    def save(self, path: str):
        """
        Save the model.

        Args:
            path (str): The pathname where the model will be saved.
        """
        with open(path, "wb") as file:
            pickle.dump(self.__dict__, file)

    @classmethod
    def load(cls, path: str):
        """
        Load the model.

        Args:
            path (str): The pathname where the model is saved.
        """
        try:
            with open(path, "rb") as file:
                model = cls.__new__(cls)
                model.__dict__ = pickle.load(file)

                return model
        except FileNotFoundError:
            print(f"File '{path}' not found.")
        except EOFError:
            print(f"Error while reading '{path}'. File may be corrupted.")