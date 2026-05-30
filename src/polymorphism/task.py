"""Bounded-depth Dyck-3 task with depth labels.

Tokens:
  PAD=0, BOS=1, EOS=2, '('=3, ')'=4, '['=5, ']'=6, '{'=7, '}'=8
  (vocab indices 9..39 are unused; vocab_size=40 per directive.)

Per-position outputs:
  - bracket-type-id: the index of the bracket-token actually at this position
    (PAD/BOS/EOS map to PAD's class for the cross-entropy mask, which is
    ignored during loss; the *meaningful* labels are the bracket-type ids).
  - current-depth: number of unmatched opens after consuming this position
    (clamped to MAX_DEPTH=8). 0 if invalid-from-the-start.
  - sticky-invalid-flag: 0 until the sequence first becomes invalid; 1 from
    that point onward (sticky).

A sequence is "valid" iff every prefix has non-negative depth and never
exceeds MAX_DEPTH, and closing brackets match the most recent open
(by family). Invalid sequences are still labeled position-by-position;
once invalid, depth is frozen at the value at the moment of failure.

Implementation notes:
  - `label_sequence_np` is a tight numpy implementation; ~50x faster than
    pure-Python torch indexing. Used inside sample_batch.
  - `sample_batch` generates content lengths and bracket-tokens via numpy
    and only crosses to torch for the final tensor returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Token IDs
PAD = 0
BOS = 1
EOS = 2
LPAREN = 3
RPAREN = 4
LBRACK = 5
RBRACK = 6
LBRACE = 7
RBRACE = 8

OPENERS = (LPAREN, LBRACK, LBRACE)
CLOSERS = (RPAREN, RBRACK, RBRACE)
OPEN_TO_CLOSE = {LPAREN: RPAREN, LBRACK: RBRACK, LBRACE: RBRACE}
CLOSE_TO_OPEN = {RPAREN: LPAREN, RBRACK: LBRACK, RBRACE: LBRACE}

VOCAB_SIZE = 40
MAX_DEPTH = 8
N_DEPTH = MAX_DEPTH + 1     # 0..8 inclusive
N_VALID = 2


@dataclass
class TaskConfig:
    n_ctx: int = 64
    vocab_size: int = VOCAB_SIZE
    max_depth: int = MAX_DEPTH
    # Probability of generating a fully-valid sequence vs. an arbitrary one.
    p_valid: float = 0.5
    # Within a valid sequence, probability of opening vs. closing at each step.
    p_open: float = 0.55


# ---------- numpy-level labeling ----------
def label_sequence_np(tokens_np: np.ndarray) -> dict[str, np.ndarray]:
    """Vectorised-as-much-as-possible numpy labeller.
    tokens_np: [B, T] int64
    Returns 'tok', 'depth', 'valid', 'mask' as numpy arrays."""
    B, T = tokens_np.shape
    depth = np.zeros((B, T), dtype=np.int64)
    valid_flag = np.zeros((B, T), dtype=np.int64)
    mask = (tokens_np != PAD)

    # Per-row stack tracking. Avoid Python overhead by working on a numpy
    # stack and using local refs.
    OPENERS_arr = np.array(OPENERS, dtype=np.int64)
    CLOSERS_arr = np.array(CLOSERS, dtype=np.int64)
    is_open = np.isin(tokens_np, OPENERS_arr)
    is_close = np.isin(tokens_np, CLOSERS_arr)
    # family[t] is in {0,1,2} for paren/brack/brace, irrespective of open/close
    # idx of opener: ('(',3)=>0, ('[',5)=>1, ('{',7)=>2
    # idx of closer: (')',4)=>0, (']',6)=>1, ('}',8)=>2
    family = (tokens_np - 3) // 2
    family[~(is_open | is_close)] = -1

    # Allocate a stack of size n_ctx per row
    stack = np.full((B, T + 1), -1, dtype=np.int64)
    stack_top = np.zeros(B, dtype=np.int64)
    invalid = np.zeros(B, dtype=bool)
    last_depth = np.zeros(B, dtype=np.int64)

    for t in range(T):
        tok = tokens_np[:, t]
        # not-yet-invalid rows
        live = ~invalid

        # Open
        op = is_open[:, t] & live
        too_deep = op & (stack_top >= MAX_DEPTH)
        invalid[too_deep] = True
        good_open = op & ~too_deep
        # push family
        idx_b = np.flatnonzero(good_open)
        stack[idx_b, stack_top[idx_b]] = family[idx_b, t]
        stack_top[good_open] += 1

        # Close
        cl = is_close[:, t] & live
        empty = cl & (stack_top == 0)
        invalid[empty] = True
        good_close_candidate = cl & ~empty
        # check top family matches
        if good_close_candidate.any():
            idx_b = np.flatnonzero(good_close_candidate)
            top_fam = stack[idx_b, stack_top[idx_b] - 1]
            mismatch_mask = top_fam != family[idx_b, t]
            mismatch_b = idx_b[mismatch_mask]
            invalid[mismatch_b] = True
            match_b = idx_b[~mismatch_mask]
            stack_top[match_b] -= 1

        depth[:, t] = np.where(invalid, last_depth, stack_top)
        last_depth = np.where(invalid, last_depth, stack_top)
        valid_flag[:, t] = invalid.astype(np.int64)

    return {"tok": tokens_np, "depth": depth, "valid": valid_flag, "mask": mask}


def label_sequence(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """Wrapper that accepts torch tensors and returns torch tensors."""
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    device = tokens.device
    np_in = tokens.detach().cpu().numpy()
    out_np = label_sequence_np(np_in)
    return {
        "tok": torch.from_numpy(out_np["tok"]).to(device).long(),
        "depth": torch.from_numpy(out_np["depth"]).to(device).long(),
        "valid": torch.from_numpy(out_np["valid"]).to(device).long(),
        "mask": torch.from_numpy(out_np["mask"]).to(device).bool(),
    }


# ---------- numpy-level sequence generation ----------
def _gen_valid_seq_np(rng: np.random.Generator, length: int, p_open: float,
                       max_depth: int) -> np.ndarray:
    """Generate a Dyck-3 valid bracket sequence of `length` content tokens."""
    out = np.zeros(length, dtype=np.int64)
    stack = np.zeros(length + 1, dtype=np.int64)
    top = 0
    for i in range(length):
        can_open = top < max_depth
        can_close = top > 0
        if can_open and (not can_close or rng.random() < p_open):
            fam = int(rng.integers(0, 3))
            opener = OPENERS[fam]
            out[i] = opener
            stack[top] = fam
            top += 1
        elif can_close:
            top -= 1
            fam = stack[top]
            out[i] = CLOSERS[fam]
        else:
            fam = int(rng.integers(0, 3))
            opener = OPENERS[fam]
            out[i] = opener
            stack[top] = fam
            top += 1
    return out


def _gen_random_brackets_np(rng: np.random.Generator, length: int) -> np.ndarray:
    return rng.integers(LPAREN, RBRACE + 1, size=length, dtype=np.int64)


def sample_batch(
    batch_size: int,
    cfg: TaskConfig | None = None,
    rng: torch.Generator | np.random.Generator | None = None,
    length_range: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Sample a training batch (BOS + content + EOS, PAD-padded to n_ctx).

    Accepts either a torch.Generator (which we seed numpy off of) or a
    np.random.Generator. Returns torch tensors (cpu)."""
    if cfg is None:
        cfg = TaskConfig()
    if length_range is None:
        length_range = (2, cfg.n_ctx - 2)
    if rng is None:
        np_rng = np.random.default_rng()
    elif isinstance(rng, torch.Generator):
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        np_rng = np.random.default_rng(seed)
    else:
        np_rng = rng
    lo, hi = length_range
    T = cfg.n_ctx

    tokens = np.full((batch_size, T), PAD, dtype=np.int64)
    lengths = np_rng.integers(lo, hi + 1, size=batch_size)
    valid_mask = np_rng.random(batch_size) < cfg.p_valid
    for b in range(batch_size):
        L = int(lengths[b])
        if valid_mask[b]:
            content = _gen_valid_seq_np(np_rng, L, cfg.p_open, cfg.max_depth)
        else:
            content = _gen_random_brackets_np(np_rng, L)
        tokens[b, 0] = BOS
        tokens[b, 1 : 1 + L] = content
        if 1 + L < T:
            tokens[b, 1 + L] = EOS

    out = label_sequence_np(tokens)
    return {k: torch.from_numpy(v) for k, v in out.items()}


def sample_compositional_test(batch_size: int, cfg: TaskConfig | None = None,
                              rng=None) -> dict[str, torch.Tensor]:
    """Compositional held-out: deep nests of a single bracket family surrounded
    by shallow alternations of mixed families."""
    if cfg is None:
        cfg = TaskConfig()
    if rng is None or isinstance(rng, torch.Generator):
        seed = (int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
                if isinstance(rng, torch.Generator) else None)
        np_rng = np.random.default_rng(seed)
    else:
        np_rng = rng
    T = cfg.n_ctx
    tokens = np.full((batch_size, T), PAD, dtype=np.int64)

    for b in range(batch_size):
        depth = int(np_rng.integers(3, cfg.max_depth + 1))
        fam = int(np_rng.integers(0, 3))
        opener = OPENERS[fam]; closer = OPEN_TO_CLOSE[opener]
        nest = [opener] * depth + [closer] * depth
        budget = T - 2 - len(nest)
        shallow = []
        while len(shallow) + 2 <= budget:
            f = int(np_rng.integers(0, 3))
            shallow += [OPENERS[f], CLOSERS[f]]
        half = len(shallow) // 2
        # interleave: half of shallow, nest, other half (both half-counts must be even)
        # ensure even, drop one pair if needed
        half = half - (half % 2)
        left = shallow[:half]; right = shallow[half : half + (len(shallow) - half - (len(shallow) - half) % 2)]
        content = left + nest + right
        seq = [BOS] + content + [EOS]
        seq = seq[:T]
        tokens[b, : len(seq)] = np.array(seq, dtype=np.int64)

    out = label_sequence_np(tokens)
    return {k: torch.from_numpy(v) for k, v in out.items()}


def sample_long_test(batch_size: int, cfg: TaskConfig | None = None,
                     rng=None, length_range: tuple[int, int] = (50, 60)
                     ) -> dict[str, torch.Tensor]:
    """Length-generalisation test (50-60), model trained on 2-48."""
    return sample_batch(batch_size, cfg, rng, length_range=length_range)
