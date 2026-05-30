"""Tests for the Dyck-3 task generation and labeling."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from polymorphism.task import (
    BOS,
    EOS,
    LBRACK,
    LPAREN,
    MAX_DEPTH,
    RBRACK,
    RPAREN,
    TaskConfig,
    label_sequence,
    sample_batch,
    sample_compositional_test,
    sample_long_test,
)


def test_label_simple_valid():
    seq = torch.tensor([[BOS, LPAREN, LBRACK, RBRACK, RPAREN, EOS]])
    out = label_sequence(seq)
    # depths: BOS=0, '(': 1, '[': 2, ']': 1, ')': 0, EOS: 0
    expected = torch.tensor([[0, 1, 2, 1, 0, 0]])
    assert (out["depth"] == expected).all()
    assert (out["valid"] == 0).all()


def test_label_invalid_then_sticky():
    # ')' with empty stack -> invalid; rest of sequence stays invalid
    seq = torch.tensor([[BOS, RPAREN, LPAREN, RPAREN, EOS]])
    out = label_sequence(seq)
    valid = out["valid"]
    assert valid[0, 0] == 0 and valid[0, 1] == 1 and valid[0, -1] == 1
    # Once invalid the depth should freeze (no further updates)
    assert (out["depth"][0, 1:] == out["depth"][0, 1]).all()


def test_label_mismatched_bracket():
    seq = torch.tensor([[BOS, LPAREN, RBRACK, EOS]])
    out = label_sequence(seq)
    # '[' closes with '(' open => mismatch => invalid from position 2
    assert out["valid"][0, 0] == 0
    assert out["valid"][0, 1] == 0
    assert out["valid"][0, 2] == 1


def test_sample_batch_dims():
    cfg = TaskConfig(n_ctx=32)
    rng = torch.Generator(); rng.manual_seed(0)
    batch = sample_batch(7, cfg, rng, length_range=(2, 20))
    for k in ("tok", "depth", "valid", "mask"):
        assert batch[k].shape == (7, 32)
    # All depths in valid range
    assert (batch["depth"] >= 0).all() and (batch["depth"] <= MAX_DEPTH).all()


def test_compositional_and_long_distributions_produce_data():
    cfg = TaskConfig(n_ctx=64)
    rng = torch.Generator(); rng.manual_seed(0)
    b1 = sample_compositional_test(5, cfg, rng)
    b2 = sample_long_test(5, cfg, rng)
    assert b1["tok"].shape == (5, 64)
    assert b2["tok"].shape == (5, 64)


def test_depth_never_exceeds_max():
    cfg = TaskConfig(n_ctx=64, max_depth=MAX_DEPTH)
    rng = torch.Generator(); rng.manual_seed(99)
    for _ in range(20):
        b = sample_batch(32, cfg, rng)
        assert b["depth"].max().item() <= MAX_DEPTH
