"""Tests for the transformer model and its forward pass."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polymorphism.model import Config, make_model


def test_param_count():
    """Verify parameter count is in the 100k-200k band."""
    model = make_model(Config())
    n = model.num_parameters()
    assert 100_000 <= n <= 200_000, f"Expected 100k-200k params, got {n}"


def test_forward_shapes():
    cfg = Config()
    model = make_model(cfg, seed=0)
    B, T = 3, 16
    tokens = torch.randint(0, cfg.vocab_size, (B, T))
    out = model(tokens)
    assert out["tok"].shape == (B, T, cfg.vocab_size)
    assert out["depth"].shape == (B, T, cfg.n_depth)
    assert out["valid"].shape == (B, T, cfg.n_valid)


def test_forward_internals_present():
    cfg = Config()
    model = make_model(cfg, seed=1)
    B, T = 2, 8
    tokens = torch.randint(0, cfg.vocab_size, (B, T))
    out, cache = model(tokens, return_internals=True)
    assert "blocks" in cache and len(cache["blocks"]) == cfg.n_layers
    for blk_cache in cache["blocks"]:
        for k in ("resid_pre", "ln1_out", "attn_q", "attn_k", "attn_v",
                  "attn_attn", "attn_out_per_head", "resid_mid", "attn_out",
                  "ln2_out", "mlp_pre", "mlp_post", "mlp_out", "resid_post"):
            assert k in blk_cache, f"missing internal: {k}"


def test_no_biases():
    """Verify there are no bias parameters anywhere."""
    model = make_model(Config())
    for name, p in model.named_parameters():
        assert "bias" not in name, f"unexpected bias param: {name}"


def test_only_relu_activation():
    """Verify the only activation used is ReLU (no GeLU / SiLU)."""
    import inspect

    from polymorphism.model import MLP
    src = inspect.getsource(MLP.forward)
    assert "relu" in src.lower()
    assert "gelu" not in src.lower() and "silu" not in src.lower() and "gegl" not in src.lower()


def test_causal_mask_blocks_future():
    """A change at position T-1 must not affect logits at any position < T-1."""
    cfg = Config(n_ctx=12)
    model = make_model(cfg, seed=42).eval()
    tokens1 = torch.randint(0, cfg.vocab_size, (1, cfg.n_ctx))
    tokens2 = tokens1.clone(); tokens2[0, -1] = (tokens2[0, -1] + 7) % cfg.vocab_size
    with torch.no_grad():
        o1 = model(tokens1); o2 = model(tokens2)
    diff = (o1["tok"][0, :-1] - o2["tok"][0, :-1]).abs().max().item()
    assert diff < 1e-5, f"causal mask leaked: max diff = {diff}"
