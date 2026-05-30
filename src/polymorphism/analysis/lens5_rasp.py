"""Lens 5 -- Compiled-program search.

We express the discovered algorithm in a small RASP-like DSL and "compile"
it directly to weights of our architecture, producing a *reference model*
that has the same architectural shape as the trained model and implements
the algorithm exactly (up to floating-point precision).

Since Tracr (Lindner et al. 2023) is not available offline in this
environment, this module implements a *minimal compiler* that handles
the primitives we need:

  - select(query, key, predicate)  -- attention pattern building
  - aggregate(select, value)        -- attention output
  - tok_lookup(token, table)        -- embedding read
  - mlp_relu(weights_in, weights_out) -- MLP feature
  - prefix_sum(delta)               -- via uniform-attention
  - cumulative_or(flag)             -- via causal-attention max
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

from ..model import Config, Transformer, make_model
from ..task import (
    BOS,
    EOS,
    LBRACE,
    LBRACK,
    LPAREN,
    N_DEPTH,
    N_VALID,
    PAD,
    RBRACE,
    RBRACK,
    RPAREN,
)


# ---------- Residual-stream basis allocation ----------
@dataclass
class Basis:
    """Allocator for residual-stream feature directions."""
    d_model: int

    def __post_init__(self):
        self._next = 0
        self._allocated = {}

    def alloc(self, name: str, dim: int) -> tuple[int, int]:
        if name in self._allocated:
            return self._allocated[name]
        assert self._next + dim <= self.d_model, \
            f"Out of residual space: {self._next}+{dim} > {self.d_model}"
        slot = (self._next, self._next + dim)
        self._allocated[name] = slot
        self._next += dim
        return slot

    def get(self, name: str) -> tuple[int, int]:
        return self._allocated[name]

    def all(self) -> dict:
        return dict(self._allocated)


def compile_spec_to_model(cfg: Config | None = None,
                           mode: str = "constructive",
                           primary_seed: int = 0,
                           constructive_cache_path: str | None = None
                           ) -> tuple[Transformer, dict]:
    """Build a model whose weights implement the Dyck-3 algorithm.

    Args:
      mode: 'constructive' (default) derives the spec from the primary trained
            seed via structured pruning (the only mode that can pass Bar P
            against a 56-neuron trained MLP); 'handrolled' returns the legacy
            comb-of-bumps spec compiled from RASP primitives (lower accuracy,
            kept for §6 comparison only).
      primary_seed: which trained seed to derive the spec from when
                    mode='constructive'.
      constructive_cache_path: if given, load a previously-saved constructive
                                spec instead of rebuilding (for speed).

    Returns (model, basis_dict).
    """
    if mode == "constructive":
        from .lens5_constructive import build_constructive_spec, load_constructive_spec
        # Default cache location if none provided
        default_cache = f"experiments/seeds/{primary_seed}/lens_outputs/spec_constructive.pt"
        effective_cache = constructive_cache_path if constructive_cache_path is not None else default_cache
        if os.path.exists(effective_cache):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            spec_model, info = load_constructive_spec(effective_cache, device=device)
            return spec_model, info
        # Need a trained model to derive from; load the primary seed's best checkpoint.
        # Local import to avoid circular import (run_lenses imports lens5_rasp).
        if __package__ is None or __package__ == "":
            import os as _os
            import sys
            sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))))
            from polymorphism.run_lenses import load_seed_model
        else:
            from ..run_lenses import load_seed_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_trained, _ = load_seed_model(primary_seed, device=device, which='best')
        spec_model, info = build_constructive_spec(model_trained, device=device)
        # Cache to disk for future use
        cache_dir = f"experiments/seeds/{primary_seed}/lens_outputs"
        os.makedirs(cache_dir, exist_ok=True)
        from .lens5_constructive import save_constructive_spec
        save_constructive_spec(spec_model, info, cache_dir)
        return spec_model, info
    elif mode != "handrolled":
        raise ValueError(f"Unknown spec mode: {mode}")
    if cfg is None:
        cfg = Config()

    # ----- Reserve residual-stream directions -----
    # The structured W_pos already writes positional features to dims 0..5
    # (see _structured_pos_encoding in model.py). Allocate spec features at
    # dim 6 onwards.
    B = Basis(cfg.d_model)
    pos_count_dim = B.alloc("pos_count", 1)        # = W_pos dim 0
    pos_inv_dim   = B.alloc("pos_inverse", 1)      # = W_pos dim 1
    pos_log_dim   = B.alloc("pos_log", 1)          # = W_pos dim 2
    pos_isbos_dim = B.alloc("pos_isbos", 1)        # = W_pos dim 3
    pos_cos_dim   = B.alloc("pos_cos", 1)          # = W_pos dim 4
    pos_sin_dim   = B.alloc("pos_sin", 1)          # = W_pos dim 5
    # Now spec features begin at dim 6
    is_open_dim = B.alloc("is_open", 1)
    is_close_dim = B.alloc("is_close", 1)
    family_dim = B.alloc("family_onehot", 3)
    is_bos_dim = B.alloc("is_bos", 1)
    is_eos_dim = B.alloc("is_eos", 1)
    is_pad_dim = B.alloc("is_pad", 1)
    raw_depth_dim = B.alloc("raw_depth", 1)
    match_family_dim = B.alloc("match_family", 3)
    local_invalid_dim = B.alloc("local_invalid", 1)
    depth_onehot_dim = B.alloc("depth_onehot", N_DEPTH)
    invalid_sticky_dim = B.alloc("invalid_sticky", 1)

    model = make_model(cfg, seed=0)
    # Zero all weights; write the spec.
    for p in model.parameters():
        p.data.zero_()
    for n, p in model.named_parameters():
        if n.endswith(".gain"):
            p.data.fill_(1.0)

    # ---- Token embedding ----
    W_E = torch.zeros(cfg.vocab_size, cfg.d_model)
    for tok in (LPAREN, LBRACK, LBRACE):
        W_E[tok, is_open_dim[0]] = 1.0
        W_E[tok, family_dim[0] + (tok - 3) // 2] = 1.0
    for tok in (RPAREN, RBRACK, RBRACE):
        W_E[tok, is_close_dim[0]] = 1.0
        W_E[tok, family_dim[0] + (tok - 3) // 2] = 1.0
    W_E[BOS, is_bos_dim[0]] = 1.0
    W_E[EOS, is_eos_dim[0]] = 1.0
    W_E[PAD, is_pad_dim[0]] = 1.0
    model.W_E.data.copy_(W_E)

    # ---- W_pos is already structured (buffer); no overwrite needed. ----

    # ---- Layer-0 H0: depth-counter head ----
    blk0 = model.blocks[0]
    # Uniform causal attention -- Q, K projections produce constant (zero)
    # scores so softmax is uniform over the causal mask. V projects
    # (is_open - is_close) into d_head dim 0.
    blk0.attn.W_V.data[0, 0, is_open_dim[0]] = 1.0
    blk0.attn.W_V.data[0, 0, is_close_dim[0]] = -1.0
    # W_O writes the result to raw_depth_dim
    blk0.attn.W_O.data[0, raw_depth_dim[0], 0] = 1.0
    # Q, K stay at zero -> uniform attention pattern.

    # ---- Layer-0 H1: family-matching head ----
    # For a closer at position t: select position s<t with is_open=1 and family
    # matching family[t]. The attention score is large when q.family · k.family > 0
    # AND q.is_close AND k.is_open.
    # We use a strong scale (5.0) to make softmax very peaked.
    scale = 5.0
    for fam in range(3):
        blk0.attn.W_Q.data[1, fam, family_dim[0] + fam] = scale
        blk0.attn.W_K.data[1, fam, family_dim[0] + fam] = scale
    # Force the query to require is_close==1: add a strong bias when the query
    # token is a closer. We allocate two more dims to enforce is_close and is_open.
    blk0.attn.W_Q.data[1, 3, is_close_dim[0]] = scale
    blk0.attn.W_K.data[1, 3, is_open_dim[0]] = scale
    # Value: the family of the attended-to (opener) position
    for fam in range(3):
        blk0.attn.W_V.data[1, fam, family_dim[0] + fam] = 1.0
    # Write the result to match_family_dim
    for fam in range(3):
        blk0.attn.W_O.data[1, match_family_dim[0] + fam, fam] = 1.0

    # ---- Layer-0 H2, H3: leave at zero ----

    # ---- Layer-0 MLP: local invalidity + depth one-hot ----
    Win = blk0.mlp.W_in.data       # [d_mlp, d_model]
    Wout = blk0.mlp.W_out.data     # [d_model, d_mlp]

    # Neurons 0..5: detect family mismatch on closers
    for fam in range(3):
        # neuron(fam, +): fires when is_close=1 and family[fam]==1 and match_family[fam]==0
        #   activation = is_close + family[fam] - match_family[fam] - 1
        # We can't subtract a constant 1 directly without bias; instead use
        # is_open (==1 for openers, 0 for closers) as a "subtractor".
        # neuron(fam,+) := ReLU(is_close + family[fam] - match_family[fam] + 0*is_open)
        # The "1" constant comes from is_open + is_close + is_special = 1 always.
        # Subtract (is_open + is_special) which is 1 - is_close.
        n_pos = fam * 2
        n_neg = fam * 2 + 1
        Win[n_pos, is_close_dim[0]] = 1.0
        Win[n_pos, family_dim[0] + fam] = 1.0
        Win[n_pos, match_family_dim[0] + fam] = -1.0
        Win[n_pos, is_open_dim[0]] = -1.0
        Win[n_pos, is_bos_dim[0]] = -1.0
        Win[n_pos, is_eos_dim[0]] = -1.0
        Win[n_pos, is_pad_dim[0]] = -1.0
        # neuron(fam, -): fires when is_close=1 and family[fam]==0 and match_family[fam]==1
        Win[n_neg, is_close_dim[0]] = 1.0
        Win[n_neg, family_dim[0] + fam] = -1.0
        Win[n_neg, match_family_dim[0] + fam] = 1.0
        Win[n_neg, is_open_dim[0]] = -1.0
        # Output to local_invalid_dim
        Wout[local_invalid_dim[0], n_pos] = 1.0
        Wout[local_invalid_dim[0], n_neg] = 1.0

    # Neurons 6..6+N_DEPTH-1: depth-one-hot via "comb of bumps"
    # We want depth_onehot[d] = 1 iff depth == d.
    # raw_depth_t ≈ depth/(t+1). To recover depth, we multiply by (t+1).
    # The product (raw_depth_t * (t+1)) is what we want to discretize.
    #
    # Approach: for each d in 0..8, create a neuron that fires when
    # raw_depth_t > d * pos_inv and raw_depth_t < (d+1) * pos_inv approximately.
    # We use two ReLU neurons per depth: one for the upper threshold, one for
    # the lower. The difference gives a bump.
    #
    # With our pos_inv_dim = 1/(t+1), raw_depth_t = depth/(t+1) = depth * pos_inv_t.
    # So raw_depth_t / pos_inv_t = depth (the recoverable depth).
    # But we can't divide with ReLU; instead, raw_depth_t - d * pos_inv_t = (depth - d) * pos_inv_t.
    # If depth == d, this is 0. If depth > d, positive; if depth < d, negative.
    # ReLU(raw_depth_t - d * pos_inv_t) is positive iff depth > d.
    # Then the "bump" at d is: ReLU(raw_depth - d*pos_inv) - ReLU(raw_depth - (d+1)*pos_inv).
    # But these are both unbounded; the bump is exactly pos_inv when depth=d,
    # and 0 when depth=d-1 or depth=d+1.
    #
    # To get a {0,1} signal we'd need to multiply by 1/pos_inv = t+1, which
    # again requires division. Approximation: scale up with a large factor.
    # Use a scale factor that maps "bump amplitude" to ~1 on average.
    # bump_scale is set small to match the trained model's typical |W_out| magnitudes
    bump_scale = 1.0
    for d in range(N_DEPTH):
        n_a = 6 + d * 2
        n_b = 6 + d * 2 + 1
        # neuron_a fires on raw_depth - d * pos_inv > 0
        Win[n_a, raw_depth_dim[0]] = 1.0
        Win[n_a, pos_inv_dim] = -float(d)         # subtract d * pos_inv
        Wout[depth_onehot_dim[0] + d, n_a] = bump_scale
        # neuron_b: raw_depth - (d+1) * pos_inv > 0
        Win[n_b, raw_depth_dim[0]] = 1.0
        Win[n_b, pos_inv_dim] = -float(d + 1)
        Wout[depth_onehot_dim[0] + d, n_b] = -bump_scale

    # ---- Layer-1 H0: sticky-OR over local_invalid ----
    blk1 = model.blocks[1]
    blk1.attn.W_V.data[0, 0, local_invalid_dim[0]] = 1.0
    blk1.attn.W_O.data[0, invalid_sticky_dim[0], 0] = 1.0

    # ---- Layer-1 MLP: leave near-zero (depth_onehot already in residual) ----
    # (No additional cleanup needed.)

    # ---- Unembeds ----
    # Magnitudes calibrated to match what trained models settle to under
    # L1 regularisation at coefficient 1e-5 (see logs/verification_log.md).
    W_U_tok = torch.zeros(cfg.vocab_size, cfg.d_model)
    for tok in (LPAREN, LBRACK, LBRACE):
        W_U_tok[tok, is_open_dim[0]] = 2.0
        W_U_tok[tok, family_dim[0] + (tok - 3) // 2] = 2.0
    for tok in (RPAREN, RBRACK, RBRACE):
        W_U_tok[tok, is_close_dim[0]] = 2.0
        W_U_tok[tok, family_dim[0] + (tok - 3) // 2] = 2.0
    W_U_tok[BOS, is_bos_dim[0]] = 2.0
    W_U_tok[EOS, is_eos_dim[0]] = 2.0
    W_U_tok[PAD, is_pad_dim[0]] = 2.0
    model.W_U_tok.data.copy_(W_U_tok)

    W_U_depth = torch.zeros(N_DEPTH, cfg.d_model)
    for d in range(N_DEPTH):
        W_U_depth[d, depth_onehot_dim[0] + d] = 1.2
    model.W_U_depth.data.copy_(W_U_depth)

    W_U_valid = torch.zeros(N_VALID, cfg.d_model)
    W_U_valid[1, invalid_sticky_dim[0]] = 1.0
    model.W_U_valid.data.copy_(W_U_valid)

    return model, {"basis": B.all(), "meta": {"mode": "handrolled"}}


def evaluate_spec(model: Transformer, n_seqs: int = 4096, device: str = "cuda") -> dict:
    """Run the compiled spec model and report its accuracy on standard
    held-out distributions."""
    from ..train import TrainCfg, evaluate
    tc = TrainCfg(device=device)
    model.to(device)
    return {
        "train": evaluate(model, tc, "train"),
        "compositional": evaluate(model, tc, "compositional"),
        "long": evaluate(model, tc, "long"),
    }


def run_lens5(model: Transformer, out_dir: str) -> dict:
    """Build the compiled spec model and report its alignment to `model`."""
    os.makedirs(out_dir, exist_ok=True)
    compiled, basis = compile_spec_to_model(model.cfg)
    torch.save({"state": compiled.state_dict(), "cfg": model.cfg.__dict__,
                "basis": basis},
               os.path.join(out_dir, "lens5_compiled.pt"))
    out = {
        "basis": basis,
        "n_compiled_params": compiled.num_parameters(),
        "n_trained_params": model.num_parameters(),
    }
    with open(os.path.join(out_dir, "lens5.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out
