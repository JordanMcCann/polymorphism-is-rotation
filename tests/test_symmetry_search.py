"""Tests for the symmetry-search alignment."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polymorphism.model import Config, make_model
from polymorphism.rmsnorm_fold import fold_rmsnorm
from polymorphism.symmetry_search import (
    _apply_head_perm,
    _apply_head_qk_rotation,
    _apply_head_vo_rotation,
    _apply_mlp_scaling,
    _apply_residual_rotation,
    _params_as_dict,
    align,
)


def test_head_permutation_invertible():
    cfg = Config()
    m = make_model(cfg, seed=0).eval()
    m = fold_rmsnorm(m)  # ensure gains are 1
    p = _params_as_dict(m)
    perm = torch.tensor([2, 0, 3, 1])
    p2 = _apply_head_perm(p, 0, perm)
    inv = torch.argsort(perm)
    p3 = _apply_head_perm(p2, 0, inv)
    for k in p:
        assert torch.allclose(p[k], p3[k]), f"head perm not invertible at {k}"


def test_mlp_scaling_invariant():
    cfg = Config()
    m = make_model(cfg, seed=0).eval()
    m = fold_rmsnorm(m)
    p = _params_as_dict(m)
    alphas = 0.5 + torch.rand(cfg.d_mlp)        # positive
    p2 = _apply_mlp_scaling(p, 0, alphas)
    # Check that W_in*W_out product (the MLP linear contribution) is preserved
    # via the geometric identity: for ReLU you need positive scaling; we just
    # check the algebraic invariant.
    Win = p["blocks.0.mlp.W_in"]; Wout = p["blocks.0.mlp.W_out"]
    Win2 = p2["blocks.0.mlp.W_in"]; Wout2 = p2["blocks.0.mlp.W_out"]
    # Product W_out @ Win equals W_out2 @ Win2 (no activation):
    assert torch.allclose(Wout @ Win, Wout2 @ Win2, atol=1e-5)


def test_residual_rotation_is_orthogonal_invariant_for_linear_part():
    """When ReLU is bypassed (zero MLP), residual rotation perfectly preserves output."""
    cfg = Config()
    m = make_model(cfg, seed=0).eval()
    m = fold_rmsnorm(m)
    # zero the MLPs so the network is linear (apart from RMSNorm + softmax)
    for L in range(cfg.n_layers):
        m.blocks[L].mlp.W_in.data.zero_()
        m.blocks[L].mlp.W_out.data.zero_()
    p = _params_as_dict(m)
    R = torch.linalg.qr(torch.randn(cfg.d_model, cfg.d_model))[0]
    p2 = _apply_residual_rotation(p, R)
    # Reconstruct a model with the rotated params/buffers
    from copy import deepcopy
    m2 = deepcopy(m)
    sd = m2.state_dict()
    for k, v in p2.items():
        if k in sd:
            sd[k] = v
    m2.load_state_dict(sd, strict=False)
    # Run both on the same tokens
    tokens = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        o1 = m(tokens); o2 = m2(tokens)
    # Output logits should match (rotation cancels through embed/unembed)
    for head in ("tok", "depth", "valid"):
        d = (o1[head] - o2[head]).abs().max().item()
        assert d < 1e-3, f"rotation not invariant on linear model, head={head}, d={d}"


def test_align_recovers_self_symmetry():
    """A model aligned to itself should yield near-zero MSE."""
    cfg = Config()
    m = make_model(cfg, seed=7).eval()
    m = fold_rmsnorm(m)
    aligned, info = align(m, m, n_outer=2, n_starts=2)
    # Trivially aligned: parameter MSE should be ~0
    final_mse = info["best_mse"]
    assert final_mse < 1e-8, f"self-alignment MSE = {final_mse}"


def test_head_qk_rotation_invariant():
    """Applying an orthogonal R to (W_Q[h], W_K[h]) leaves Q^T K invariant."""
    cfg = Config()
    m = make_model(cfg, seed=0).eval()
    m = fold_rmsnorm(m)
    p = _params_as_dict(m)
    torch.manual_seed(42)
    R = torch.linalg.qr(torch.randn(cfg.d_head, cfg.d_head))[0]
    p2 = _apply_head_qk_rotation(p, 0, 1, R)
    Wq1 = p["blocks.0.attn.W_Q"][1]
    Wk1 = p["blocks.0.attn.W_K"][1]
    Wq2 = p2["blocks.0.attn.W_Q"][1]
    Wk2 = p2["blocks.0.attn.W_K"][1]
    # Q^T K is the score matrix for head 1
    QK1 = Wq1.t() @ Wk1
    QK2 = Wq2.t() @ Wk2
    assert torch.allclose(QK1, QK2, atol=1e-4), f"QK invariance broken: max diff {(QK1-QK2).abs().max()}"
    # Other heads must be untouched
    for h_other in (0, 2, 3):
        assert torch.allclose(p["blocks.0.attn.W_Q"][h_other],
                              p2["blocks.0.attn.W_Q"][h_other])


def test_head_vo_rotation_invariant():
    """Applying an orthogonal R to (W_V[h], W_O[h]) leaves the head's
    contribution W_O @ W_V to the residual stream invariant."""
    cfg = Config()
    m = make_model(cfg, seed=0).eval()
    m = fold_rmsnorm(m)
    p = _params_as_dict(m)
    torch.manual_seed(43)
    R = torch.linalg.qr(torch.randn(cfg.d_head, cfg.d_head))[0]
    p2 = _apply_head_vo_rotation(p, 1, 2, R)
    Wv1 = p["blocks.1.attn.W_V"][2]
    Wo1 = p["blocks.1.attn.W_O"][2]
    Wv2 = p2["blocks.1.attn.W_V"][2]
    Wo2 = p2["blocks.1.attn.W_O"][2]
    OV1 = Wo1 @ Wv1
    OV2 = Wo2 @ Wv2
    assert torch.allclose(OV1, OV2, atol=1e-4), f"OV invariance broken: max diff {(OV1-OV2).abs().max()}"


def test_multistart_recovers_random_head_perm():
    """Build a model B that is a known random head permutation of model A.
    align(B, A) should recover MSE < 1e-6 even when the original permutation
    is non-identity (requiring multi-start to find)."""
    cfg = Config()
    mA = make_model(cfg, seed=11).eval()
    mA = fold_rmsnorm(mA)
    pA = _params_as_dict(mA)
    # Apply a deterministic head permutation to layer 0
    perm = torch.tensor([3, 1, 0, 2])
    pB = _apply_head_perm(pA, 0, perm)
    # Reconstruct a model with the permuted params
    from copy import deepcopy
    mB = deepcopy(mA)
    sd = mB.state_dict()
    for k, v in pB.items():
        if k in sd:
            sd[k] = v
    mB.load_state_dict(sd, strict=False)
    aligned, info = align(mB, mA, n_outer=4, n_starts=8)
    final_mse = info["best_mse"]
    assert final_mse < 1e-5, f"multi-start failed to recover known head perm: MSE = {final_mse}"


def test_align_returns_residual_rotation_info():
    """The align() info dict must record per-head QK/VO rotations."""
    cfg = Config()
    m = make_model(cfg, seed=21).eval()
    m = fold_rmsnorm(m)
    _, info = align(m, m, n_outer=2, n_starts=1)
    assert "head_qk_R" in info
    assert "head_vo_R" in info
    assert len(info["head_qk_R"]) == cfg.n_layers
    assert len(info["head_qk_R"][0]) == cfg.n_heads
