from typing import Optional

import pickle

import torch
import torch.nn as nn

from tokenizer.tokenizer import Tokenizer

# star wars files
# save / load model
# doc strings

class LanguageModel():
    def __init__(
        self,
        tokenizer: Tokenizer,
        embed_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        forward_expansion: int,
        dropout: float,
        device: str = 'cpu'
    ):
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
            dropout
        ).to(device)

    def predict(self, input_text: str, max_new_tokens: int) -> str:

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
        with open(path, "wb") as file:
            pickle.dump(self.__dict__, file)
            # torch.save(self.encoder.state_dict(), './model_weights.pt')

    @classmethod
    def load(cls, path: str):
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

class Encoder(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embed_size: int, 
        context_length: int, 
        num_layers: int, 
        num_heads: int, 
        forward_expansion: int, 
        dropout: float
    ):
        super().__init__()
        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(context_length, embed_size)

        self.blocks = nn.Sequential(*[TransformerBlock(num_heads, embed_size, context_length, forward_expansion, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.linear_head = nn.Linear(embed_size, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, 
        idx: torch.tensor, # (B, T)
        targets: Optional[torch.tensor] = None # (B, T)
    ):
        B, T = idx.shape

        token_embedding = self.token_embedding(idx) # (B, T, es)
        position_embedding = self.position_embedding(torch.arange(T)) # (T, es)

        x = token_embedding + position_embedding # (B, T, es)
        x = self.blocks(x) # (B, T, es)
        x = self.layer_norm(x) # (B, T, es)
        logits = self.linear_head(x) # (B, T, vs)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B * T, C) # (B * T, vs)
            targets = targets.view(B * T) # (B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

class TransformerBlock(nn.Module):
    def __init__(
        self, 
        num_heads: int, 
        embed_size: int, 
        context_length: int, 
        forward_expansion: int, 
        dropout: float
    ):
        super().__init__()
        assert embed_size % num_heads == 0, 'embed_size not divisible by num_heads'

        head_size = embed_size // num_heads
        self.attention = MultiHeadAttention(num_heads, head_size, embed_size, context_length, dropout)

        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.tensor # (B, T, es)
    ) -> torch.tensor:
        attn = self.attention(x) # (B, T, es)

        x = self.dropout(self.layer_norm_1(attn + x)) # (B, T, es)

        forward = self.feed_forward(x) # (B, T, es)

        x = self.dropout(self.layer_norm_2(forward + x)) # (B, T, es)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        num_heads: int, 
        head_size: int, 
        embed_size: int, 
        context_length: int, 
        dropout: float
    ):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(head_size, embed_size, context_length, dropout) for _ in range(num_heads)
        ])
        
        self.proj = nn.Linear(head_size * num_heads, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.tensor # (B, T, es)
    ) -> torch.tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, es)
        out = self.dropout(self.proj(out)) # (B, T, es)
        return out

class AttentionHead(nn.Module):
    def __init__(
        self, 
        head_size: int, 
        embed_size: int, 
        context_length: int, 
        dropout: float
    ):
        super().__init__()
        self.key_proj = nn.Linear(embed_size, head_size, bias=False)
        self.query_proj = nn.Linear(embed_size, head_size, bias=False)
        self.value_proj = nn.Linear(embed_size, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.tensor # (B, T, es)
    ) -> torch.tensor:
        B, T, C = x.shape

        k = self.key_proj(x) # (B, T, hs)
        q = self.query_proj(x) # (B, T, hs)
        v = self.value_proj(x) # (B, T, hs)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf')) # (B, T, T)
        wei = nn.functional.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out