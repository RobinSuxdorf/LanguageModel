from typing import Self
from dataclasses import dataclass

import torch
import torch.nn as nn

from tokenizers import bpe_tokenizer, special_tokens, tokenizer
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
        device (torch.device): The device where the calculations will be done, i.e. CPU or CUDA.
        args (ModelArgs): The model hyperparameters.
    """
    def __init__(
        self,
        tokenizer: tokenizer.Tokenizer = bpe_tokenizer.BytePairEncodingTokenizer.read_pkl('./tokenizers/trained_tokenizers/bpe.pkl'),
        device: torch.device = torch.device('cpu'),
        args: ModelArgs = ModelArgs()
    ) -> None:
        """
        Initialize the language model.

        Args:
            tokenizer (Tokenizer): The tokenizer for encoding and decoding the text.
            device (torch.device): The device where the calculations will be done, i.e. CPU or CUDA.
            args (ModelArgs): The model hyperparameters.
        """
        self.tokenizer = tokenizer

        self._excluded_tokens: list[int] = [
            self.tokenizer.stoi[special_tokens.SpecialTokens.SOS],
            self.tokenizer.stoi[special_tokens.SpecialTokens.PAD]
        ]

        self.context_length = args.context_length
        self._device = device

        self.encoder = model.Encoder(
            vocab_size = len(tokenizer),
            embed_size  = args.embed_size,
            context_length = args.context_length,
            num_layers = args.num_layers,
            num_heads = args.num_heads,
            forward_expansion = args.forward_expansion,
            dropout = args.dropout,
            device = device
        ).to(device)

    def predict(self, input_text: str, max_new_tokens: int = 100) -> str:
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

            logits = self.encoder(idx_cond)

            logits = logits[:, -1, :]

            probs = nn.functional.softmax(logits, dim=-1)

            # exclude specific tokens from the choice
            for token_id in self._excluded_tokens:
                probs[:, token_id] = 0.0

            probs /= probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next.item() == self.tokenizer.stoi[special_tokens.SpecialTokens.EOS]:
                break

            context = torch.cat((context, idx_next), dim=1)

        output = self.tokenizer.decode(context[0].tolist())
        return output

    def save(self, path: str) -> None:
        """
        Save the model.

        Args:
            path (str): The pathname where the model will be saved.
        """
        torch.save({
            "tokenizer": self.tokenizer,
            "context_length": self.context_length,
            "model_state_dict": self.encoder.state_dict()
        }, path)

    @classmethod
    def load(cls, path: str, device: torch.device) -> Self:
        """
        Load the model.

        Args:
            path (str): The pathname where the model is saved.
            device (torch.device): The device on which the model will be loaded.
        """
        try:
            checkpoint = torch.load(path, map_location=device)

            model = cls.__new__(cls)
            model.__init__(
                tokenizer=checkpoint['tokenizer'],
                device=device,
                args=ModelArgs(context_length=checkpoint['context_length'])
            )
            model.encoder.load_state_dict(checkpoint['model_state_dict'])

            return model

        except FileNotFoundError:
            print(f"File '{path}' not found.")
            
        except EOFError:
            print(f"Error while reading '{path}'. File may be corrupted.")