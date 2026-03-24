from enum import Enum

import torch
from torch import Tensor
from torch.nn import GELU, Dropout, LayerNorm, Linear, Module, functional


class AttentionType(Enum):
    CAUSAL = 1
    AUTOREGRESSIVE = 2
    UNCONSTRAINED = 3
    CONVOLUTIONAL = 4


class SelfAttention(Module):
    def __init__(self, n_head: int, n_embd: int, bias: bool, dropout: float, is_causal: bool):
        super().__init__()

        assert n_embd % n_head == 0

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.is_causal = is_causal

        # key, query, value projections for all heads, but in a batch
        self.c_attn = Linear(n_embd, 3 * n_embd, bias=bias)

        # output projection
        self.c_proj = Linear(n_embd, n_embd, bias=bias)

        # regularization
        self.resid_dropout = Dropout(dropout)

    def forward(self, x):
        batch_size, length, num_features = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(batch_size, length, self.n_head, num_features // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(batch_size, length, self.n_head, num_features // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(batch_size, length, self.n_head, num_features // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )

        y = y.transpose(1, 2)
        y = torch.reshape(y, (batch_size, length, num_features))

        # output projection
        y = self.c_proj(y)
        return self.resid_dropout(y)


class AutoregressiveSelfAttention(Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        n_embd_side: int,
        bias: bool,
        dropout: float,
        n_embd_kq: int = 192,
    ):
        super().__init__()

        assert n_embd % n_head == 0
        assert n_embd_side % n_head == 0
        assert n_embd_kq % n_head == 0

        n_embd_out = n_embd + n_embd_side

        self.n_head = n_head
        self.n_embd = n_embd
        self.n_embd_out = n_embd_out
        self.n_embd_side = n_embd_side
        self.dropout = dropout
        self.n_embd_kq = n_embd_kq

        # key, query, value projections for all heads, but in a batch
        self.c_attn_q = Linear(self.n_embd_side, n_embd_kq, bias=bias)
        self.c_attn_kv = Linear(n_embd_out, n_embd_kq + n_embd_out, bias=bias)

        # output projection
        self.c_proj = Linear(n_embd_out, n_embd_out, bias=bias)

        # regularization
        self.resid_dropout = Dropout(dropout)

        # first embeddings
        self.embedder = Linear(self.n_embd_side, n_embd_out, bias=bias)

    def forward(self, x: Tensor, side: Tensor) -> Tensor:
        batch_size, length, _ = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.c_attn_q(side)
        q = q.view(batch_size, length, self.n_head, self.n_embd_kq // self.n_head)
        q = torch.transpose(q, 1, 2)  # (B, nh, T, hs)

        kv = torch.cat((x, side), dim=2)
        kv = self.c_attn_kv(kv)

        k = kv[:, :, : self.n_embd_kq]
        k = k.view(batch_size, length, self.n_head, self.n_embd_kq // self.n_head)
        k = torch.transpose(k, 1, 2)  # (B, nh, T, hs)

        v = kv[:, :, self.n_embd_kq :]
        v = v.view(batch_size, length, self.n_head, self.n_embd_out // self.n_head)
        v = torch.transpose(v, 1, 2)  # (B, nh, T, hs)

        first_embeddings = self.embedder(side[:, 0])
        first_embeddings = first_embeddings.view(batch_size, self.n_head, 1, self.n_embd_out // self.n_head)

        if length > 1:
            attn_mask = torch.ones((length, length), dtype=torch.bool, device=x.device)
            attn_mask = torch.tril(attn_mask, diagonal=-1)
            attn_mask = torch.cat((attn_mask[[1]], attn_mask[1:]), dim=0)

            # causal self-attention; self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )

            y = torch.cat((first_embeddings, y[:, :, 1:]), dim=2)

        else:
            y = first_embeddings

        y = torch.transpose(y, 1, 2)
        y = torch.reshape(y, (batch_size, length, self.n_embd_out))

        # output projection
        y = self.c_proj(y)
        return self.resid_dropout(y)


class ConvolutionalSelfAttention(Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float, length: int = 5):
        super().__init__()

        self.n_embd = n_embd
        self.dropout = dropout
        self.length = length

        self.c_attn = Linear(n_embd, n_embd, bias=bias)

        # output projection
        self.c_proj = Linear(n_embd, n_embd, bias=bias)

        # regularization
        self.resid_dropout = Dropout(dropout)

    def forward(self, x):
        batch_size, length, num_features = x.shape

        v = self.c_attn(x)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        length_masked = self.length // 2 + 1

        weight = torch.zeros(size=(num_features, num_features, self.length), dtype=v.dtype, device=v.device)
        weight[:, :, :length_masked] = 1 / length_masked

        y = functional.conv1d(v, weight, padding='same')
        y = y.transpose(1, 2)
        y = torch.reshape(y, (batch_size, length, num_features))

        # output projection
        y = self.c_proj(y)
        return self.resid_dropout(y)


class MLP(Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float):
        super().__init__()

        self.c_fc = Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = GELU()
        self.c_proj = Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        bias: bool,
        dropout: float,
        n_embd_out: int | None = None,
        attention_type: AttentionType = AttentionType.CAUSAL,
    ):
        super().__init__()

        self.n_embd = n_embd
        self.attention_type = attention_type

        n_embd_out = n_embd_out or n_embd

        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.proj = Linear(n_embd, n_embd_out, bias=bias) if n_embd != n_embd_out else None
        self.ln_2 = LayerNorm(n_embd_out, bias=bias)
        self.mlp = MLP(n_embd_out, bias, dropout)

        if attention_type == AttentionType.CAUSAL:
            self.attn = SelfAttention(n_head, n_embd, bias, dropout, is_causal=True)
        elif attention_type == AttentionType.UNCONSTRAINED:
            self.attn = SelfAttention(n_head, n_embd, bias, dropout, is_causal=False)
        else:
            self.attn = ConvolutionalSelfAttention(n_embd, bias, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = self.proj(x) if self.proj else x
        return x + self.mlp(self.ln_2(x))


class BlockAutoregressive(Module):
    def __init__(
        self,
        n_head: int,
        n_embd: int,
        n_embd_side: int,
        bias: bool,
        dropout: float,
    ):
        super().__init__()

        n_embd_out = n_embd + n_embd_side

        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = AutoregressiveSelfAttention(n_head, n_embd, n_embd_side, bias, dropout)
        self.ln_2 = LayerNorm(n_embd_out, bias=bias)
        self.mlp = MLP(n_embd_out, bias, dropout)

    def forward(self, x: Tensor, side: Tensor) -> Tensor:
        x = self.attn(self.ln_1(x), side)
        return x + self.mlp(self.ln_2(x))
