import torch
import torch.nn.functional as F
from torch import nn


def alibi_slopes(num_heads: int) -> torch.Tensor:
    # Simple monotonic slopes are enough here: each head gets a different distance penalty.
    return torch.logspace(0, -3, steps=num_heads, base=2.0)


class AlibiSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.output = nn.Linear(embedding_dim, embedding_dim)
        self.register_buffer("slopes", alibi_slopes(num_heads).view(1, num_heads, 1, 1))

    def build_alibi_mask(self, sequence_length: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(sequence_length, device=device)
        distances = positions.view(sequence_length, 1) - positions.view(1, sequence_length)
        causal_mask = distances < 0
        bias = -self.slopes.to(device) * distances.clamp_min(0).view(1, 1, sequence_length, sequence_length)
        return bias.masked_fill(causal_mask.view(1, 1, sequence_length, sequence_length), float("-inf"))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = hidden.shape
        qkv = self.qkv(hidden).view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = self.build_alibi_mask(sequence_length, hidden.device)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = AlibiSelfAttention(embedding_dim, num_heads)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, embedding_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        hidden = hidden + self.feedforward(self.feedforward_norm(hidden))
        return hidden


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        max_sequence_length: int,
        feedforward_dim: int | None = None,
    ) -> None:
        super().__init__()
        feedforward_dim = feedforward_dim or embedding_dim * 4
        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim, num_heads, feedforward_dim)
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _, sequence_length = input_ids.shape
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max_sequence_length "
                f"{self.max_sequence_length}"
            )

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.output_norm(hidden))
