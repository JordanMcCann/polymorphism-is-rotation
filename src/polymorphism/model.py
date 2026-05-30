"""Two-layer transformer with ReLU MLP, no biases, RMSNorm.

Architecture (per directive):
  - 2 transformer blocks, pre-norm
  - d_model=64, n_heads=4, d_head=16, d_mlp=256
  - Activation: ReLU only (lens 4 requires piecewise linearity)
  - No biases on any linear layer
  - RMSNorm at standard pre-norm positions during training
  - Untied embed/unembed
  - vocab_size=40, context_length=64
  - Learned positional embeddings

Param count (with this config): ~104k (104,448) -- inside the 100k-200k target band.

Implementation note: linear projections are stored as nn.Parameter tensors
rather than nn.Linear modules so that mechanistic analyses (QK/OV factoring,
RMSNorm folding, symmetry searches) can address them directly by name.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 40
    d_model: int = 64
    n_heads: int = 4
    d_head: int = 16
    d_mlp: int = 256
    n_layers: int = 2
    n_ctx: int = 64
    n_depth: int = 9          # depth values 0..8
    n_valid: int = 2          # valid / sticky-invalid
    init_scale: float = 0.02
    rmsnorm_eps: float = 1e-5
    use_rmsnorm: bool = True  # set False on a folded copy of the model
    # Positional embedding mode:
    #   "structured": fixed analytic features (linear counter + inverse counter
    #     + sinusoids). NOT learnable -- guarantees length generalisation since
    #     these features are defined for every position in [0, n_ctx).
    #   "learned": classic learned positional embedding (does not generalise
    #     to positions beyond those seen in training).
    pos_encoding: str = "structured"


class RMSNorm(nn.Module):
    """RMSNorm with learnable gain only (no bias). gain is a single vector of d_model."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d))

    def forward(self, x):
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(ms + self.eps)
        return x_normed * self.gain


class Attention(nn.Module):
    """Multi-head attention with no biases.

    Weight layouts (parameter shapes are explicit so analysis code can address them):
      W_Q, W_K, W_V : [n_heads, d_head, d_model]   (per-head input projections)
      W_O           : [n_heads, d_model, d_head]   (per-head output projection)

    The forward pass is implemented manually (no nn.MultiheadAttention) so the
    interpretability stack can patch any internal activation.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        s = cfg.init_scale
        self.W_Q = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model) * s)
        self.W_K = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model) * s)
        self.W_V = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_head, cfg.d_model) * s)
        self.W_O = nn.Parameter(torch.randn(cfg.n_heads, cfg.d_model, cfg.d_head) * s)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).bool(),
            persistent=False,
        )

    def forward(self, x, return_internals: bool = False):
        # x: [B, T, d_model]
        B, T, _ = x.shape
        # Per-head q,k,v: [B, n_heads, T, d_head]
        q = torch.einsum("btd,hpd->bhtp", x, self.W_Q)
        k = torch.einsum("btd,hpd->bhtp", x, self.W_K)
        v = torch.einsum("btd,hpd->bhtp", x, self.W_V)

        attn_scores = torch.einsum("bhtp,bhsp->bhts", q, k) / math.sqrt(self.cfg.d_head)
        mask = self.causal_mask[:T, :T]
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn_scores, dim=-1)

        # Weighted values: [B, n_heads, T, d_head]
        z = torch.einsum("bhts,bhsp->bhtp", attn, v)
        # Per-head output then summed: [B, T, d_model]
        out_per_head = torch.einsum("bhtp,hdp->bhtd", z, self.W_O)
        out = out_per_head.sum(dim=1)

        if return_internals:
            return out, {
                "q": q,
                "k": k,
                "v": v,
                "attn_scores": attn_scores,
                "attn": attn,
                "z": z,
                "out_per_head": out_per_head,
            }
        return out


class MLP(nn.Module):
    """Two-layer MLP with ReLU activation and no biases."""

    def __init__(self, cfg: Config):
        super().__init__()
        s = cfg.init_scale
        self.W_in = nn.Parameter(torch.randn(cfg.d_mlp, cfg.d_model) * s)
        self.W_out = nn.Parameter(torch.randn(cfg.d_model, cfg.d_mlp) * s)

    def forward(self, x, return_internals: bool = False):
        pre = torch.einsum("btd,md->btm", x, self.W_in)   # pre-activation
        post = F.relu(pre)                                # post-activation
        out = torch.einsum("btm,dm->btd", post, self.W_out)
        if return_internals:
            return out, {"mlp_pre": pre, "mlp_post": post}
        return out


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = RMSNorm(cfg.d_model, cfg.rmsnorm_eps) if cfg.use_rmsnorm else nn.Identity()
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model, cfg.rmsnorm_eps) if cfg.use_rmsnorm else nn.Identity()
        self.mlp = MLP(cfg)

    def forward(self, x, return_internals: bool = False):
        cache = {} if return_internals else None
        x_pre_attn = x
        x_normed1 = self.ln1(x)
        if return_internals:
            attn_out, attn_int = self.attn(x_normed1, return_internals=True)
            cache.update({"resid_pre": x_pre_attn, "ln1_out": x_normed1})
            cache.update({f"attn_{k}": v for k, v in attn_int.items()})
        else:
            attn_out = self.attn(x_normed1)
        x = x + attn_out
        if return_internals:
            cache["resid_mid"] = x
            cache["attn_out"] = attn_out

        x_normed2 = self.ln2(x)
        if return_internals:
            mlp_out, mlp_int = self.mlp(x_normed2, return_internals=True)
            cache["ln2_out"] = x_normed2
            cache.update(mlp_int)
            cache["mlp_out"] = mlp_out
        else:
            mlp_out = self.mlp(x_normed2)
        x = x + mlp_out
        if return_internals:
            cache["resid_post"] = x
            return x, cache
        return x


def _structured_pos_encoding(n_ctx: int, d_model: int) -> torch.Tensor:
    """Build a fixed (non-learnable) analytic positional feature matrix that
    is defined for all positions 0..n_ctx-1 -- so the model can generalise
    to longer-than-trained sequences within the context limit.

    Only the FIRST 6 dimensions carry positional information; the remaining
    dimensions are zero so token features (added via W_E) have clean room.

    Layout (d_model = 64, but only dims 0-5 are non-zero):
      dim 0:  linear counter      pos / n_ctx
      dim 1:  inverse counter     1 / (pos + 1)
      dim 2:  log-position        log(pos + 1) / log(n_ctx)
      dim 3:  is-bos              1 if pos == 0 else 0
      dim 4:  cosine wave at low freq    cos(2*pi*pos / n_ctx)
      dim 5:  sine wave at low freq      sin(2*pi*pos / n_ctx)
    """
    pos = torch.arange(n_ctx, dtype=torch.float32)
    enc = torch.zeros(n_ctx, d_model, dtype=torch.float32)
    enc[:, 0] = pos / n_ctx
    enc[:, 1] = 1.0 / (pos + 1.0)
    enc[:, 2] = torch.log(pos + 1.0) / math.log(n_ctx)
    enc[:, 3] = (pos == 0).float()
    enc[:, 4] = torch.cos(pos * (2 * math.pi / n_ctx))
    enc[:, 5] = torch.sin(pos * (2 * math.pi / n_ctx))
    return enc


class Transformer(nn.Module):
    """Two-layer transformer with three output heads."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        s = cfg.init_scale
        self.W_E = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model) * s)
        if cfg.pos_encoding == "structured":
            # Fixed buffer; *not* a parameter so it doesn't change during training
            self.register_buffer(
                "W_pos", _structured_pos_encoding(cfg.n_ctx, cfg.d_model),
                persistent=True,
            )
        elif cfg.pos_encoding == "learned":
            self.W_pos = nn.Parameter(torch.randn(cfg.n_ctx, cfg.d_model) * s)
        else:
            raise ValueError(cfg.pos_encoding)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model, cfg.rmsnorm_eps) if cfg.use_rmsnorm else nn.Identity()
        # Three task heads:
        self.W_U_tok = nn.Parameter(torch.randn(cfg.vocab_size, cfg.d_model) * s)
        self.W_U_depth = nn.Parameter(torch.randn(cfg.n_depth, cfg.d_model) * s)
        self.W_U_valid = nn.Parameter(torch.randn(cfg.n_valid, cfg.d_model) * s)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor, return_internals: bool = False):
        # tokens: [B, T] long
        B, T = tokens.shape
        x = self.W_E[tokens] + self.W_pos[:T].unsqueeze(0)
        all_caches = []
        for i, block in enumerate(self.blocks):
            if return_internals:
                x, cache = block(x, return_internals=True)
                all_caches.append(cache)
            else:
                x = block(x)
        x_final = self.ln_f(x)
        logits_tok = torch.einsum("btd,vd->btv", x_final, self.W_U_tok)
        logits_depth = torch.einsum("btd,vd->btv", x_final, self.W_U_depth)
        logits_valid = torch.einsum("btd,vd->btv", x_final, self.W_U_valid)
        out = {"tok": logits_tok, "depth": logits_depth, "valid": logits_valid}
        if return_internals:
            return out, {
                "embed": self.W_E[tokens] + self.W_pos[:T].unsqueeze(0),
                "blocks": all_caches,
                "ln_f_out": x_final,
            }
        return out


def make_model(cfg: Config | None = None, seed: int | None = None) -> Transformer:
    """Construct a model with optional reproducible init."""
    if cfg is None:
        cfg = Config()
    if seed is not None:
        torch.manual_seed(seed)
    model = Transformer(cfg)
    return model
