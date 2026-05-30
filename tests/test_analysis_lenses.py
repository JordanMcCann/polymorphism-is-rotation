"""End-to-end tests for the analysis lenses on a freshly-initialised model."""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from polymorphism.analysis.lens1_weights import run_lens1
from polymorphism.analysis.lens3_causal import (
    attribution_patch,
    mean_activations,
)
from polymorphism.analysis.lens4_polyhedral import run_lens4
from polymorphism.analysis.lens5_rasp import compile_spec_to_model
from polymorphism.model import Config, make_model
from polymorphism.task import TaskConfig, sample_batch


def test_lens1_runs():
    m = make_model(Config(), seed=0).eval()
    with tempfile.TemporaryDirectory() as td:
        out = run_lens1(m, td)
        assert len(out["heads"]) == m.cfg.n_layers * m.cfg.n_heads
        assert len(out["mlps"]) == m.cfg.n_layers
        for h in out["heads"]:
            assert "qk" in h and "ov" in h
            assert h["qk"]["M_norm"] >= 0
            assert h["ov"]["Z_norm"] >= 0


def test_lens3_mean_activations_runs():
    if not torch.cuda.is_available():
        return
    m = make_model(Config(), seed=0).eval().cuda()
    means = mean_activations(m, n_seqs=64, device="cuda")
    # one entry per component
    expected = []
    for L in range(m.cfg.n_layers):
        for h in range(m.cfg.n_heads):
            expected.append(f"attn_{L}_h{h}")
        expected.append(f"mlp_{L}")
    for k in expected:
        assert k in means
        assert means[k].shape == (m.cfg.d_model,)


def test_lens3_attribution_signs():
    if not torch.cuda.is_available():
        return
    m = make_model(Config(), seed=0).eval().cuda()
    means = mean_activations(m, n_seqs=64, device="cuda")
    rng = np.random.default_rng(0)
    batch = sample_batch(64, TaskConfig(n_ctx=m.cfg.n_ctx), rng, length_range=(2, 30))
    batch = {k: v.cuda() for k, v in batch.items()}
    attr = attribution_patch(m, batch, means)
    # No NaNs (all components have a defined predicted effect)
    for k, v in attr["predicted_effects"].items():
        assert not (v != v), f"NaN in attribution for {k}"


def test_lens4_polyhedra_runs():
    if not torch.cuda.is_available():
        return
    m = make_model(Config(), seed=0).eval().cuda()
    with tempfile.TemporaryDirectory() as td:
        out = run_lens4(m, td, n_seqs=128, device="cuda")
        assert len(out["layers"]) == m.cfg.n_layers
        for L in out["layers"]:
            assert L["n_unique_patterns"] >= 1
            assert L["n_samples"] > 0


def test_lens5_compile_runs():
    cfg = Config()
    # Use handrolled mode for this test (constructive mode requires a trained
    # checkpoint, which is created by the integration pipeline).
    m, info = compile_spec_to_model(cfg, mode="handrolled")
    assert m.num_parameters() == cfg.vocab_size * cfg.d_model * 2 \
        + cfg.n_layers * (3 * cfg.n_heads * cfg.d_head * cfg.d_model
                           + cfg.n_heads * cfg.d_model * cfg.d_head
                           + cfg.d_mlp * cfg.d_model + cfg.d_model * cfg.d_mlp
                           + 2 * cfg.d_model) \
        + cfg.d_model \
        + cfg.n_depth * cfg.d_model + cfg.n_valid * cfg.d_model
    # basis (under handrolled) should contain expected names
    basis = info["basis"]
    expected = ["is_open", "is_close", "family_onehot", "depth_onehot",
                 "invalid_sticky", "local_invalid"]
    for n in expected:
        assert n in basis, f"missing basis name: {n}"
