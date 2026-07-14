import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import rearrange


def init_linear_weights(
    in_dim: int,
    out_dim: int,
    device: str | None = None,
    dtype=None,
):
    var = 2 / (in_dim + out_dim)
    std = var**0.5
    data = torch.empty(out_dim, in_dim, device=device, dtype=dtype)
    nn.init.trunc_normal_(data, 0, std, -3 * std, 3 * std)
    return data


def init_embedding_weights(
    num_embeddings: int,
    embedding_dim: int,
    device: str | None = None,
    dtype=None,
):
    data = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
    nn.init.trunc_normal_(data, 0, 1, -3, 3)
    return data


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        data = init_linear_weights(in_features, out_features, device, dtype)
        self.weight = nn.Parameter(data)

    def forward(self, x):
        return torch.einsum("...i, oi-> ...o", x, self.weight)


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        data = init_embedding_weights(num_embeddings, embedding_dim, device, dtype)
        self.embeds = nn.Parameter(data).to(device)

    def forward(self, x):
        return self.embeds[x]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.gain = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype),
        )
        self.eps = eps

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        denom = torch.sqrt(
            torch.square(x).mean(
                dim=-1,
                keepdim=True,
            )
            + self.eps
        )
        # print(denom.shape)
        result = x / denom * self.gain

        return result.to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        inner = silu(self.w1(x)) * self.w3(x)
        return self.w2(inner)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        k_vec = torch.arange(1, d_k // 2 + 1, dtype=torch.float32, device=device)
        exponent = (2 * k_vec - 2) / self.d_k
        inv_freq = 1 / theta**exponent
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.einsum("i,k -> ik", positions, inv_freq)
        self.cos_vals = torch.cos(angles)
        self.sin_vals = torch.sin(angles)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_k"],
        token_positions: Float[Tensor, " ... seq_len"],
    ):
        cos = self.cos_vals[token_positions]
        sin = self.sin_vals[token_positions]

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        out = torch.stack([x1_rot, x2_rot], dim=-1).flatten(start_dim=-2)
        return out


def scaled_dot_product_attention(
    q: Float[Tensor, "batch ... seq_len d_k"],
    k: Float[Tensor, "batch ... seq_len d_k"],
    v: Float[Tensor, "batch ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"],
):
    d_model = q.shape[-1]
    q_k = torch.einsum("b ... i d, b ... j d -> b ... i j", q, k) / d_model**0.5
    q_k = q_k.masked_fill(~mask, float("-inf"))
    att_weights = softmax(q_k)
    return att_weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_head = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        # Having a single Q linear layer is the same as having N heads each
        # projecting to a smaller dimension, reason being is that its still
        # a matrix multiply and the rows
        self.Q = Linear(d_model, d_model, device, dtype)
        self.K = Linear(d_model, d_model, device, dtype)
        self.V = Linear(d_model, d_model, device, dtype)
        self.O = Linear(d_model, d_model, device, dtype)

    def forward(self, x):
        seq_len = x.shape[1]
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        res = scaled_dot_product_attention(q, k, v, mask)
        res = rearrange(res, "b h s d -> b s (h d)", h=self.num_heads)
        return self.O(res)


class MultiHeadAttentionRope(MultiHeadAttention):
    def __init__(
        self,
        d_model,
        num_heads,
        theta,
        max_seq_len,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device)

    def forward(self, x, positions):
        seq_len = x.shape[1]
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

        res = scaled_dot_product_attention(q, k, v, mask)
        res = rearrange(res, "b h s d -> b s (h d)", h=self.num_heads)
        return self.O(res)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.att_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ff_norm = RMSNorm(d_model, device=device, dtype=dtype)

        self.mha = MultiHeadAttentionRope(
            d_model,
            num_heads,
            theta,
            max_seq_len,
            device=device,
            dtype=dtype,
        )
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        seq_len = x.shape[1]
        token_pos = torch.arange(0, seq_len)

        att_out = self.mha(self.att_norm(x), token_pos)
        x = x + att_out
        ff_out = self.ffn(self.ff_norm(x))
        x = x + ff_out

        return x


def softmax(x):
    max_ele, _ = x.max(dim=-1, keepdim=True)
    x -= max_ele
    return torch.exp(x) / torch.exp(x).sum(dim=-1, keepdim=True)
