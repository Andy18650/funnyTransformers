import torch
import torch.nn.functional as F
from torch import nn


class SqrtSoftplus(nn.Module):
    """sqrt(softplus(x)): smooth and positive, but keeps growing (~sqrt(x)) for
    large inputs instead of saturating, giving nonlinearity even far from 0."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(F.softplus(x))


# Plain activations, instantiated with no arguments.
ACTIVATIONS = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "softplus": nn.Softplus,
    "sqrt_softplus": SqrtSoftplus,
}


def build_activation(name: str) -> nn.Module:
    return ACTIVATIONS[name.lower()]()


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

    def forward(self, hidden: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length, embedding_dim = hidden.shape
        qkv = self.qkv(hidden).view(batch_size, sequence_length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = self.build_alibi_mask(sequence_length, hidden.device)
        if segment_ids is not None:
            # Forbid attention across document boundaries: a query may only attend to
            # keys sharing its segment id. Shapes: segment_ids (B, T) -> cross (B, 1, T, T).
            cross_document = segment_ids[:, None, :, None] != segment_ids[:, None, None, :]
            attn_mask = attn_mask.expand(batch_size, -1, -1, -1).masked_fill(
                cross_document, float("-inf")
            )
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        return self.output(attended)


class FeedForward(nn.Module):
    """Transformer FFN with an optional gated linear branch.

    gate="none":   down(act(up(x)))                  -- standard FFN
    gate="linear": down(act(up(x)) * value(x))       -- GLU variant (e.g. SwiGLU
                   when activation="silu"), where value is a parallel linear
                   up-projection with no activation.
    """

    def __init__(
        self,
        embedding_dim: int,
        feedforward_dim: int,
        activation: str = "gelu",
        gate: str = "none",
    ) -> None:
        super().__init__()
        if gate not in ("none", "linear"):
            raise ValueError(f"Unsupported gate '{gate}'. Choose from: linear, none.")
        self.gate = gate
        self.up = nn.Linear(embedding_dim, feedforward_dim)
        self.activation = build_activation(activation)
        self.value = nn.Linear(embedding_dim, feedforward_dim) if gate == "linear" else None
        self.down = nn.Linear(feedforward_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.up(x))
        if self.value is not None:
            hidden = hidden * self.value(x)
        return self.down(hidden)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        feedforward_dim: int,
        activation: str = "gelu",
        ffn_gate: str = "none",
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = AlibiSelfAttention(embedding_dim, num_heads)
        self.feedforward_norm = nn.LayerNorm(embedding_dim)
        self.feedforward = FeedForward(embedding_dim, feedforward_dim, activation, ffn_gate)

    def forward(self, hidden: torch.Tensor, segment_ids: torch.Tensor | None = None) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), segment_ids)
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
        activation: str = "gelu",
        ffn_gate: str = "none",
        bos_token_id: int | None = None,
        intra_doc_masking: bool = False,
    ) -> None:
        super().__init__()
        feedforward_dim = feedforward_dim or embedding_dim * 4
        self.max_sequence_length = max_sequence_length
        self.bos_token_id = bos_token_id
        self.intra_doc_masking = intra_doc_masking
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(embedding_dim, num_heads, feedforward_dim, activation, ffn_gate)
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

        segment_ids = None
        if self.intra_doc_masking and self.bos_token_id is not None:
            # Each <bos> opens a new document, so a running count of <bos> tokens
            # labels every position with the document it belongs to.
            segment_ids = torch.cumsum(input_ids == self.bos_token_id, dim=1)

        hidden = self.token_embedding(input_ids)
        for block in self.blocks:
            hidden = block(hidden, segment_ids)
        return self.output(self.output_norm(hidden))


# Model hyperparameters read from the flat config. Everything else in the config
# (dataset, training, logging) is ignored here.
MODEL_KEYS = (
    "embedding_dim",
    "num_layers",
    "num_heads",
    "feedforward_dim",
    "activation",
    "ffn_gate",
    "intra_doc_masking",
)


def build_transformer(config: dict, *, vocab_size: int, bos_token_id: int | None = None):
    """The single place a model is constructed from a flat config. max_sequence_length
    is derived from the training sequence_length so it never has to be set twice."""
    kwargs = {key: config[key] for key in MODEL_KEYS if key in config}
    return TransformerLanguageModel(
        vocab_size=vocab_size,
        max_sequence_length=config["sequence_length"],
        bos_token_id=bos_token_id,
        **kwargs,
    )
