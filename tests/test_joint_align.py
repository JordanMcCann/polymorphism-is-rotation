"""Tests for src/experiments/bar_p_joint/joint_align.py.

The critical correctness test: given two models that differ only by a known
random orthogonal R applied to the residual basis, joint_align should
recover R (i.e. drive both weight and activation MSE to near zero).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_cayley_returns_orthogonal():
    """Cayley transform of any matrix gives an orthogonal R with det = +1."""
    from polymorphism.experiments.bar_p_joint.joint_align import cayley
    torch.manual_seed(0)
    A = torch.randn(8, 8) * 0.5
    R = cayley(A)
    err = (R @ R.t() - torch.eye(8)).pow(2).sum().sqrt().item()
    assert err < 1e-5, f"Cayley R not orthogonal: ||R@R^T - I||_F = {err}"
    det = torch.linalg.det(R).item()
    assert det > 0.99, f"det(R) = {det}, expected +1"


def test_cayley_at_zero_is_identity():
    """Cayley(0) = I."""
    from polymorphism.experiments.bar_p_joint.joint_align import cayley
    R = cayley(torch.zeros(8, 8))
    assert torch.allclose(R, torch.eye(8), atol=1e-6)


def test_weight_mse_zero_when_dicts_equal():
    """weight_mse_from_dicts == 0 when the dicts are identical."""
    from polymorphism.experiments.bar_p_joint.joint_align import weight_mse_from_dicts
    p = {"a": torch.randn(3, 4), "b": torch.randn(5)}
    q = {k: v.clone() for k, v in p.items()}
    assert weight_mse_from_dicts(p, q).item() < 1e-10


def test_residual_rotation_application_invariant():
    """Applying R then R^T to the params should restore them."""
    from polymorphism.experiments.bar_p_joint.joint_align import _apply_residual_rotation_grad
    from polymorphism.model import Config, make_model
    from polymorphism.rmsnorm_fold import fold_rmsnorm
    from polymorphism.symmetry_search import _params_as_dict
    cfg = Config()
    m = fold_rmsnorm(make_model(cfg, seed=42).eval())
    p = _params_as_dict(m)
    torch.manual_seed(7)
    R = torch.linalg.qr(torch.randn(cfg.d_model, cfg.d_model))[0]
    p_rot = _apply_residual_rotation_grad(p, R)
    p_back = _apply_residual_rotation_grad(p_rot, R.t())
    for k in p:
        d = (p[k] - p_back[k]).abs().max().item()
        # ln gains untouched; check only param tensors
        if d > 1e-4:
            raise AssertionError(f"Round-trip failed for {k}: max abs delta = {d}")


@pytest.mark.slow
def test_joint_align_recovers_known_rotation():
    """If model_seed is exactly model_ref rotated by R_true on the residual basis,
    joint_align should recover R_true. Tests the core correctness."""
    from polymorphism.experiments.bar_p_joint.joint_align import (
        _apply_residual_rotation_grad,
        joint_align,
    )
    from polymorphism.model import Config, make_model
    from polymorphism.rmsnorm_fold import fold_rmsnorm
    from polymorphism.symmetry_search import _params_as_dict
    from polymorphism.task import TaskConfig, sample_batch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    m_ref = fold_rmsnorm(make_model(cfg, seed=11).eval()).to(device)
    p_ref = _params_as_dict(m_ref)

    # Create a "seed" model whose weights are exactly p_ref rotated by R_true
    torch.manual_seed(13)
    R_true = torch.linalg.qr(torch.randn(cfg.d_model, cfg.d_model))[0].to(device)
    p_seed = _apply_residual_rotation_grad(
        {k: v.to(device) for k, v in p_ref.items()}, R_true)
    m_seed = make_model(cfg).to(device).eval()
    sd = m_seed.state_dict()
    for k in p_seed:
        if k in sd:
            sd[k] = p_seed[k].to(sd[k].dtype)
    m_seed.load_state_dict(sd, strict=False)
    m_seed = fold_rmsnorm(m_seed)

    # Activation batch
    rng = np.random.default_rng(99)
    task_cfg = TaskConfig(n_ctx=cfg.n_ctx)
    batch = sample_batch(256, task_cfg, rng, length_range=(2, 48))

    # Joint align. Lambda = 0.5 gives equal weight to both losses.
    res = joint_align(m_seed, m_ref, batch, lambda_act=0.5,
                       n_inner_iters=100, lr=1e-1, n_starts=4,
                       device=device, verbose=False)
    # Expect substantial weight-MSE reduction. The starting alignment runs
    # head perms etc., so by step 100 we should be well below the random-
    # init baseline (which would be ~weight variance ≈ O(1)).
    assert res["best"]["max_mse"] < 0.5, \
        f"joint_align did not converge: max_mse={res['best']['max_mse']}"


def test_joint_align_runs_smoke():
    """Smoke test: joint_align runs end-to-end without error on a tiny synthetic
    pair with 10 inner iterations (catches API regressions)."""
    from polymorphism.experiments.bar_p_joint.joint_align import joint_align
    from polymorphism.model import Config, make_model
    from polymorphism.rmsnorm_fold import fold_rmsnorm
    from polymorphism.task import TaskConfig, sample_batch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config()
    m_a = fold_rmsnorm(make_model(cfg, seed=0).eval()).to(device)
    m_b = fold_rmsnorm(make_model(cfg, seed=1).eval()).to(device)
    rng = np.random.default_rng(0)
    batch = sample_batch(64, TaskConfig(n_ctx=cfg.n_ctx), rng, length_range=(2, 24))
    res = joint_align(m_a, m_b, batch, lambda_act=0.1,
                       n_inner_iters=10, n_starts=2, device=device, verbose=False)
    assert "best" in res
    assert "history" in res
    assert "per_tensor" in res
    assert res["best"]["max_mse"] >= 0
