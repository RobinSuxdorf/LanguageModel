import pickle

import torch
import torch.nn as nn

from tokenizer.tokenizer import Tokenizer
from .model import Encoder

class LanguageModel():
    """
    Language model using encoder-only design.

    Args:
        tokenizer (Tokenizer): The tokenizer for encoding and decoding the text.
        embed_size (int): The dimension of the token embedding.
        context_length (int): The amount of tokens the language model can consider for predicting the next token.
        num_layers (int): The number of transformer blocks of the model.
        num_heads (int): The number of AttentionHeads in each layer.
        forward_expansion (int): The expansion factor for the linear layer at the end of the transformer block.
        dropout (float): The dropout value for the layers.
        device (str): The device where the calculations will be done, i.e. CPU or CUDA.
    """
    def __init__(
        self,
        tokenizer: Tokenizer,
        embed_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        forward_expansion: int,
        dropout: float,
        device: str
    ):
        """
        Initialize the language model.

        Args:
            tokenizer (Tokenizer): The tokenizer for encoding and decoding the text.
            embed_size (int): The dimension of the token embedding.
            context_length (int): The amount of tokens the language model can consider for predicting the next token.
            num_layers (int): The number of transformer blocks of the model.
            num_heads (int): The number of AttentionHeads in each layer.
            forward_expansion (int): The expansion factor for the linear layer at the end of the transformer block.
            dropout (float): The dropout value for the layers.
            device (str): The device where the calculations will be done, i.e. CPU or CUDA.
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self._device = device

        self.encoder = Encoder(
            tokenizer.vocab_size,
            embed_size,
            context_length,
            num_layers,
            num_heads,
            forward_expansion,
            dropout,
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
            # torch.save(self.encoder.state_dict(), './model_weights.pt')

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
                # model.encoder.load_state_dict(torch.load('./model_weights.pt'))
                return model
        except FileNotFoundError:
            print(f"File '{path}' not found.")
        except EOFError:
            print(f"Error while reading '{path}'. File may be corrupted.")