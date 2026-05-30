"""Fold RMSNorm gains analytically into the surrounding linear weights.

For each RMSNorm layer N with learned gain `g`, the operation is
    y = (x / sqrt(mean(x^2) + eps)) * g
The next operation in the architecture is a linear projection W applied to y.
Since `g` is a constant vector applied element-wise, we have

    W @ (y) = W @ ((x / rms) * g) = (W * g[None, :]) @ (x / rms)

i.e. the gain `g` can be absorbed into the columns of W (the last axis,
which indexes d_model). After folding we set g <- ones, so the RMSNorm
becomes a pure scaling operation that depends only on the input data.

This preserves *every* forward pass exactly (up to floating-point noise).
We assert that on a random input before returning the folded model.

Args / returns:
    fold_rmsnorm(model) -> Transformer   (a fresh model, original untouched)
"""

from __future__ import annotations

import copy

import torch

from .model import Transformer


def _fold_gain_into(W: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """W has its last axis = d_model. Multiply along that axis by `gain`."""
    return W * gain.view(*([1] * (W.ndim - 1)), -1)


def fold_rmsnorm(model: Transformer) -> Transformer:
    """Return a copy of `model` with all RMSNorm gains folded into the
    weights of the next linear projection. After this call every RMSNorm
    has gain == 1, but the normalization itself remains (it depends on
    input statistics and cannot be analytically removed)."""
    folded = copy.deepcopy(model)

    for layer_idx in range(folded.cfg.n_layers):
        blk = folded.blocks[layer_idx]
        # ln1 feeds attention's W_Q, W_K, W_V
        g1 = blk.ln1.gain.data.clone()
        blk.attn.W_Q.data = _fold_gain_into(blk.attn.W_Q.data, g1)
        blk.attn.W_K.data = _fold_gain_into(blk.attn.W_K.data, g1)
        blk.attn.W_V.data = _fold_gain_into(blk.attn.W_V.data, g1)
        blk.ln1.gain.data = torch.ones_like(g1)
        # ln2 feeds the MLP's W_in
        g2 = blk.ln2.gain.data.clone()
        blk.mlp.W_in.data = _fold_gain_into(blk.mlp.W_in.data, g2)
        blk.ln2.gain.data = torch.ones_like(g2)

    # ln_f feeds all three unembed heads
    gf = folded.ln_f.gain.data.clone()
    folded.W_U_tok.data = _fold_gain_into(folded.W_U_tok.data, gf)
    folded.W_U_depth.data = _fold_gain_into(folded.W_U_depth.data, gf)
    folded.W_U_valid.data = _fold_gain_into(folded.W_U_valid.data, gf)
    folded.ln_f.gain.data = torch.ones_like(gf)

    return folded


def verify_fold(model: Transformer, folded: Transformer, n_samples: int = 8,
                atol: float = 1e-5, rtol: float = 1e-4) -> dict:
    """Run a random input through both models and assert outputs match.

    Returns a dict with the max abs differences per output head."""
    model.eval()
    folded.eval()
    torch.manual_seed(0)
    tokens = torch.randint(
        0, model.cfg.vocab_size, (n_samples, model.cfg.n_ctx), dtype=torch.long
    )
    with torch.no_grad():
        out_orig = model(tokens)
        out_fold = folded(tokens)
    diffs = {}
    for k in out_orig:
        d = (out_orig[k] - out_fold[k]).abs().max().item()
        diffs[k] = d
        assert torch.allclose(
            out_orig[k], out_fold[k], atol=atol, rtol=rtol
        ), f"Fold mismatch on output head '{k}': max abs diff = {d}"
    return diffs
