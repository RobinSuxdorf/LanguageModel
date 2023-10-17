import torch
import torch.nn as nn

from tokenizer.tokenizer import Tokenizer

## LanguageModel
## predict
## save
## load

class Encoder(nn.Module):
    def __init__(
        self, 
        tokenizer: Tokenizer, 
        embed_size: int, 
        context_length: int, 
        num_layers: int, 
        num_heads: int, 
        forward_expansion: int, 
        dropout: float
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length

        self.token_embedding = nn.Embedding(tokenizer.vocab_size, embed_size)
        self.position_embedding = nn.Embedding(context_length, embed_size)

        self.blocks = nn.Sequential(*[TransformerBlock(num_heads, embed_size, context_length, forward_expansion, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.linear_head = nn.Linear(embed_size, tokenizer.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embedding = self.token_embedding(idx)
        position_embedding = self.position_embedding(torch.arange(T))

        x = token_embedding + position_embedding
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.linear_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :]

            probs = nn.functional.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def predict(input_text: str, max_new_tokens: int) -> str:
        pass

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        attn = self.attention(x)

        x = self.dropout(self.layer_norm_1(attn + x))

        forward = self.feed_forward(x)

        x = self.dropout(self.layer_norm_2(forward + x))

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

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
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

    def forward(self, x: torch.tensor) -> torch.tensor:
        B, T, C = x.shape
        k = self.key_proj(x)
        q = self.query_proj(x)
        v = self.value_proj(x)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v

        return out