import torch
import torch.nn as nn


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
        device: str | None = None,
        dtype=None,
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
        device: str | None = None,
        dtype=None,
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
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        pass
