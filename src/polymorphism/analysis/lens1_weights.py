"""Lens 1 -- Direct weight decomposition.

Following Elhage et al. (2021), every parameter of the (folded) network
is decomposed into a small set of interpretable circuits:

  - For each attention head, the QK circuit (how the head selects which
    token to attend to) and the OV circuit (what the head writes to the
    residual stream from the attended-to token).
  - For each MLP, the SVD of W_in and W_out, the "neuron-by-input" map
    W_out @ W_in (the rank-d_model linear approximation to MLP behavior),
    and a per-neuron interpretation: which residual directions push it
    on/off, and which residual directions it writes.
  - For the embed and unembed, a residual-direction-by-token map.

The output is a structured JSON dictionary that the other lenses and the
artifact reference by name.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

from ..model import Transformer
from ..rmsnorm_fold import fold_rmsnorm


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


def head_qk_circuit(model: Transformer, layer: int, head: int) -> dict:
    """QK circuit: x_q^T (W_Q^T W_K) x_k. Compose with embeddings to get
    token-by-token attention scores (ignoring positional embedding).

    Returns: M [d_model, d_model], plus token-by-token score matrix
             [vocab, vocab]."""
    folded = fold_rmsnorm(model)
    blk = folded.blocks[layer]
    # W_Q: [n_heads, d_head, d_model]
    Wq = _np(blk.attn.W_Q[head])    # [d_head, d_model]
    Wk = _np(blk.attn.W_K[head])    # [d_head, d_model]
    M = Wq.T @ Wk                   # [d_model, d_model]
    # Compose with embeddings: x_q = W_E[v_q], x_k = W_E[v_k]
    # score[v_q, v_k] = W_E[v_q] @ M @ W_E[v_k]^T  (token-only contribution)
    W_E = _np(model.W_E)             # [V, d_model]
    tok_scores = W_E @ M @ W_E.T     # [V, V]
    return {
        "M": M.tolist(),
        "M_norm": float(np.linalg.norm(M)),
        "tok_scores": tok_scores.tolist(),
        "tok_score_top": _topk_per_row(tok_scores, k=3),
    }


def head_ov_circuit(model: Transformer, layer: int, head: int) -> dict:
    """OV circuit: what the head writes to the residual stream given the
    attended-to token. Z = W_O W_V [d_model, d_model]. Composed with embed
    and unembed (token head) it produces a token-by-token transition matrix."""
    folded = fold_rmsnorm(model)
    blk = folded.blocks[layer]
    Wv = _np(blk.attn.W_V[head])   # [d_head, d_model]
    Wo = _np(blk.attn.W_O[head])   # [d_model, d_head]
    Z = Wo @ Wv                     # [d_model, d_model]
    W_E = _np(model.W_E); W_U_tok = _np(model.W_U_tok)
    tok_to_tok = W_U_tok @ Z @ W_E.T     # [V_out, V_in]
    return {
        "Z": Z.tolist(),
        "Z_norm": float(np.linalg.norm(Z)),
        "tok_to_tok": tok_to_tok.tolist(),
        "tok_to_tok_top": _topk_per_row(tok_to_tok, k=3),
    }


def _topk_per_row(M: np.ndarray, k: int = 3) -> list:
    """Helper -- for each row return top-k columns by score."""
    top = []
    for r in range(M.shape[0]):
        idx = np.argsort(-M[r])[:k]
        top.append({"row": int(r), "top": [(int(i), float(M[r, i])) for i in idx]})
    return top


def mlp_decomposition(model: Transformer, layer: int) -> dict:
    """SVD of W_in and W_out, plus neuron-by-input read/write directions."""
    folded = fold_rmsnorm(model)
    blk = folded.blocks[layer]
    Win = _np(blk.mlp.W_in)          # [d_mlp, d_model]   -- reads from residual
    Wout = _np(blk.mlp.W_out)        # [d_model, d_mlp]   -- writes to residual

    Uw, Sw, Vw = np.linalg.svd(Win, full_matrices=False)
    Uo, So, Vo = np.linalg.svd(Wout, full_matrices=False)

    # Per-neuron: row of Win is the "input direction" for that neuron;
    # column of Wout is the "output direction" (where it writes).
    in_norms = np.linalg.norm(Win, axis=1)   # [d_mlp]
    out_norms = np.linalg.norm(Wout, axis=0) # [d_mlp]
    # Cosine between in-direction and out-direction of each neuron
    cos_per_neuron = (Win * Wout.T).sum(axis=1) / (in_norms * out_norms + 1e-9)

    # Compose with embeddings to see neuron read/write effects on tokens
    W_E = _np(model.W_E)
    W_U_tok = _np(model.W_U_tok)
    # Neuron's input-side activation strength per token
    neuron_in_per_tok = Win @ W_E.T               # [d_mlp, V]
    # Neuron's output-side direct logit effect per token (token head)
    neuron_out_per_tok = W_U_tok @ Wout            # [V, d_mlp]

    return {
        "Win_singular": Sw.tolist(),
        "Wout_singular": So.tolist(),
        "in_norms": in_norms.tolist(),
        "out_norms": out_norms.tolist(),
        "cos_per_neuron": cos_per_neuron.tolist(),
        "neuron_in_per_tok_top": _topk_per_row(neuron_in_per_tok, k=4),
        "neuron_out_per_tok_top": _topk_per_row(neuron_out_per_tok.T, k=4),
    }


def embed_unembed_summary(model: Transformer) -> dict:
    W_E = _np(model.W_E); W_U_tok = _np(model.W_U_tok)
    W_U_depth = _np(model.W_U_depth); W_U_valid = _np(model.W_U_valid)
    # SVD of W_E and each unembed
    out = {}
    for name, mat in [("W_E", W_E), ("W_U_tok", W_U_tok),
                       ("W_U_depth", W_U_depth), ("W_U_valid", W_U_valid)]:
        _, S, _ = np.linalg.svd(mat, full_matrices=False)
        out[f"{name}_singular"] = S.tolist()
    # Direct embed-unembed identity check (a "copy" path)
    out["copy_score"] = (W_U_tok @ W_E.T).diagonal().tolist()
    return out


def positional_pattern(model: Transformer, layer: int, head: int) -> dict:
    """Attention from position p to position q via positional embeddings alone."""
    folded = fold_rmsnorm(model)
    blk = folded.blocks[layer]
    Wq = _np(blk.attn.W_Q[head]); Wk = _np(blk.attn.W_K[head])
    M = Wq.T @ Wk
    W_pos = _np(model.W_pos)
    pos_scores = W_pos @ M @ W_pos.T
    return {
        "pos_scores": pos_scores.tolist(),
        "pos_score_diag_minus_1": np.diag(pos_scores, k=-1).tolist(),
        "pos_score_diag_0":       np.diag(pos_scores, k=0).tolist(),
        "pos_score_diag_minus_2": np.diag(pos_scores, k=-2).tolist(),
    }


def run_lens1(model: Transformer, out_dir: str) -> dict[str, Any]:
    """Compute all Lens-1 quantities and save to out_dir/lens1.json."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = model.cfg
    results: dict[str, Any] = {
        "embed_unembed": embed_unembed_summary(model),
        "heads": [],
        "mlps": [],
    }
    for L in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            qk = head_qk_circuit(model, L, h)
            ov = head_ov_circuit(model, L, h)
            pos = positional_pattern(model, L, h)
            results["heads"].append({
                "layer": L, "head": h,
                "qk": {"M_norm": qk["M_norm"], "tok_score_top": qk["tok_score_top"]},
                "ov": {"Z_norm": ov["Z_norm"], "tok_to_tok_top": ov["tok_to_tok_top"]},
                "pos": pos,
            })
        results["mlps"].append({
            "layer": L,
            **mlp_decomposition(model, L),
        })
    out_path = os.path.join(out_dir, "lens1.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    return results
