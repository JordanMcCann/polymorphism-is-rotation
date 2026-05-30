"""Tests for analytical RMSNorm gain folding."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polymorphism.model import Config, make_model
from polymorphism.rmsnorm_fold import fold_rmsnorm, verify_fold


def test_fold_identity_when_gains_one():
    """With gain==1 everywhere, folding is a no-op."""
    cfg = Config()
    m = make_model(cfg, seed=0).eval()
    # set all gains to 1
    for n, p in m.named_parameters():
        if n.endswith(".gain"):
            p.data.fill_(1.0)
    f = fold_rmsnorm(m)
    diffs = verify_fold(m, f, n_samples=4, atol=1e-6, rtol=1e-5)
    assert max(diffs.values()) < 1e-6


def test_fold_with_random_gains_preserves_output():
    cfg = Config()
    m = make_model(cfg, seed=1).eval()
    # randomize gains a bit (around 1)
    torch.manual_seed(3)
    for n, p in m.named_parameters():
        if n.endswith(".gain"):
            p.data = 0.5 + torch.rand_like(p.data)        # in [0.5, 1.5]
    f = fold_rmsnorm(m)
    diffs = verify_fold(m, f, n_samples=8, atol=1e-4, rtol=1e-4)
    assert max(diffs.values()) < 1e-4


def test_fold_sets_gains_to_one():
    cfg = Config()
    m = make_model(cfg, seed=2).eval()
    for n, p in m.named_parameters():
        if n.endswith(".gain"):
            p.data = torch.randn_like(p.data) * 0.3 + 1.0
    f = fold_rmsnorm(m)
    for n, p in f.named_parameters():
        if n.endswith(".gain"):
            assert torch.allclose(p.data, torch.ones_like(p.data))
