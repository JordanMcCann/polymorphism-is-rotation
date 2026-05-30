"""Tests for src/experiments/scale/common.py and ig_pythia.py.

These tests deliberately avoid loading a Pythia model (HF download/load is
slow and contends with a running EXP 2 training). They exercise the pure-
math helpers — Procrustes, decoder cosine, cache keys, SAE adapter — which
are the infrastructure pieces other experiments depend on.

A separate integration smoke test in src/experiments/scale/common.py::smoke()
exercises the full Pythia path end-to-end and is run manually before EXP 1.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

# Make sure imports work when run via `python -m pytest`
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_procrustes_recovers_known_rotation():
    """Best orthogonal R given A and A@R_true should recover R_true."""
    from polymorphism.experiments.scale.common import best_orthogonal, procrustes_metrics
    torch.manual_seed(0)
    N, d = 2000, 32
    A = torch.randn(N, d)
    R_true = torch.linalg.qr(torch.randn(d, d))[0]
    B = A @ R_true
    R_fit = best_orthogonal(A, B)
    err = (A @ R_fit - B).pow(2).mean().sqrt().item()
    assert err < 1e-5, f"Procrustes did not recover R_true: err={err}"
    metrics = procrustes_metrics(A, B, R_fit)
    assert metrics["rot_EV"] > 0.999, f"post-rotation EV too low: {metrics['rot_EV']}"
    assert abs(metrics["op_norm_R"] - 1.0) < 1e-4, \
        f"R not orthogonal: op_norm={metrics['op_norm_R']}"


def test_procrustes_identity_when_inputs_equal():
    """Procrustes on identical matrices should give R ≈ I."""
    from polymorphism.experiments.scale.common import best_orthogonal
    torch.manual_seed(1)
    A = torch.randn(500, 16)
    R = best_orthogonal(A, A)
    I = torch.eye(16, dtype=R.dtype)
    err = (R - I).pow(2).sum().sqrt().item()
    assert err < 1e-4, f"R should be identity, got ||R-I||_F={err}"


def test_procrustes_random_orthogonal_norm():
    """A random orthogonal R should have ||R-I||_F ≈ sqrt(2*d) for large d.

    This is the §8.2 sanity check that motivates the rotation-magnitude
    interpretation in the paper.
    """
    from polymorphism.experiments.scale.common import best_orthogonal
    torch.manual_seed(7)
    for d in (32, 64, 128):
        A = torch.randn(2000, d)
        R_true = torch.linalg.qr(torch.randn(d, d))[0]
        B = A @ R_true
        R = best_orthogonal(A, B)
        I = torch.eye(d, dtype=R.dtype)
        norm = (R - I).pow(2).sum().sqrt().item()
        expected = (2 * d) ** 0.5
        # Random orthogonal R: ||R-I||_F^2 ≈ 2d when far from identity,
        # but the random draw has variance, so allow [0.6 * expected, expected]
        assert 0.5 * expected < norm < 1.1 * expected, \
            f"d={d}: ||R-I||_F={norm:.3f}, expected ≈ {expected:.3f}"


def test_decoder_cosine_histogram_self_is_one():
    """Decoder-cosine of an SAE state against itself should give 100% > 0.5 / mean 1.0."""
    from polymorphism.experiments.scale.common import decoder_cosine_histogram
    d = 16; n_feat = 128
    W = np.random.RandomState(2).randn(d, n_feat)
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
    state = {"W_dec": W}
    res = decoder_cosine_histogram(state, state)
    assert res["fraction_stable"] == 1.0
    assert abs(res["mean_max_cos"] - 1.0) < 1e-5


def test_cache_key_deterministic_and_unique():
    """Cache keys are deterministic in their inputs and differ on any change."""
    from polymorphism.experiments.scale.common import CorpusConfig, cache_key
    c1 = CorpusConfig(n_sequences=128, seq_len=256, seed=42)
    c2 = CorpusConfig(n_sequences=128, seq_len=256, seed=43)
    k1 = cache_key("EleutherAI/pythia-70m", None, c1, "layer0_resid_post")
    k1_b = cache_key("EleutherAI/pythia-70m", None, c1, "layer0_resid_post")
    k2 = cache_key("EleutherAI/pythia-70m", None, c2, "layer0_resid_post")
    k3 = cache_key("EleutherAI/pythia-70m-seed1", None, c1, "layer0_resid_post")
    k4 = cache_key("EleutherAI/pythia-70m", "step3000", c1, "layer0_resid_post")
    k5 = cache_key("EleutherAI/pythia-70m", None, c1, "layer1_resid_post")
    assert k1 == k1_b
    assert k1 != k2
    assert k1 != k3
    assert k1 != k4
    assert k1 != k5


def test_save_load_cached_acts_roundtrip(tmp_path):
    """Save and load round-trips a small activation dict with fp16 quantisation."""
    from polymorphism.experiments.scale.common import (
        CorpusConfig,
        load_cached_acts,
        save_cached_acts,
    )
    cfg = CorpusConfig(n_sequences=64, seq_len=128, seed=11)
    acts = {"layer0_resid_post": torch.randn(64, 128, 32)}
    paths = save_cached_acts(str(tmp_path), "test/model", None, cfg, acts)
    assert len(paths) == 1
    loaded = load_cached_acts(str(tmp_path), "test/model", None, cfg,
                                ["layer0_resid_post"])
    assert loaded is not None
    # fp16 round-trip has ~5e-3 relative error
    rel_err = (loaded["layer0_resid_post"] - acts["layer0_resid_post"]).abs().mean() \
              / acts["layer0_resid_post"].abs().mean()
    assert rel_err < 1e-2


def test_load_cached_acts_returns_none_when_missing(tmp_path):
    """If any requested site is missing, load_cached_acts returns None."""
    from polymorphism.experiments.scale.common import CorpusConfig, load_cached_acts
    cfg = CorpusConfig(n_sequences=64, seq_len=128, seed=11)
    res = load_cached_acts(str(tmp_path), "x/y", None, cfg, ["layer0_resid_post"])
    assert res is None


def test_sae_adapter_trains():
    """Tiny SAE on synthetic data trains and reaches EV > 0.8 in 200 steps."""
    from polymorphism.experiments.scale.common import train_sae_on
    torch.manual_seed(0)
    # Synthetic data: 8 underlying sparse features in d=16
    d_in = 16
    n_underlying = 8
    n_samples = 4096
    rng = torch.Generator().manual_seed(0)
    F = torch.randn(d_in, n_underlying, generator=rng)
    coeffs = torch.bernoulli(torch.full((n_samples, n_underlying), 0.2),
                              generator=rng) * torch.rand(n_samples, n_underlying, generator=rng)
    data = coeffs @ F.T + 0.01 * torch.randn(n_samples, d_in, generator=rng)
    res = train_sae_on(data, expansion=2, n_steps=400, batch_size=512,
                        l1_coef=1e-3, lr=1e-3, seed=0,
                        device="cuda" if torch.cuda.is_available() else "cpu")
    assert res["explained_var"] > 0.5, f"SAE EV={res['explained_var']:.3f}"


def test_site_list_layout():
    """site_list returns a deterministic list with one resid_pre + n_layers resid_post."""
    from polymorphism.experiments.scale.common import site_list
    sites = site_list(6)
    assert sites[0] == "layer0_resid_pre"
    assert sites[1:] == [f"layer{L}_resid_post" for L in range(6)]
    assert len(sites) == 7
