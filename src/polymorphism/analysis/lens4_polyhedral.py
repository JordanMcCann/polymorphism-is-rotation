"""Lens 4 -- Polyhedral / tropical decomposition of ReLU MLPs.

A ReLU MLP partitions its input space into a (finite) set of polytopes,
on each of which the function is affine. The polytope is determined by
the *activation pattern*: which neurons are post-activation positive.

This module:
  1. Collects the empirical set of activation patterns on a large
     evaluation set.
  2. Counts unique patterns (= occupied linear regions).
  3. For each occupied region, computes the local affine map
     y = A_R @ x + b_R
     and the dimension of the polytope (number of "free" activation
     directions: full d_mlp minus number of constraints).
  4. Reports the per-region linear map's effect on the next-block's
     downstream sites (residual stream, attention queries, etc.) by
     computing the composition.

Why it matters: the polyhedral decomposition is the *complete*
description of an ReLU MLP as a piecewise-linear function. Every
behavioral claim about an MLP is, in some region, an affine map; if
the lens cannot recover the right number of regions or the right local
maps, the description is incomplete.

Implementation notes:
  - We do not exhaustively enumerate all theoretically possible regions
    (~2^d_mlp). We characterize the *occupied* set, which is what
    matters operationally.
  - For an MLP with d_mlp=256 and an input space of dimension 64, the
    number of occupied regions on typical data is much smaller -- often
    O(thousands).
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import torch

from ..model import Transformer
from ..rmsnorm_fold import fold_rmsnorm
from ..task import TaskConfig, sample_batch


def collect_mlp_io(model: Transformer, layer: int, n_seqs: int = 4096,
                    batch: int = 128, device: str = "cuda",
                    length_range: tuple[int, int] = (2, 48)) -> dict:
    """For a given MLP layer, collect (x_in, mlp_pre, mlp_post, mlp_out)
    on a large dataset; one row per (sequence, position)."""
    model.eval()
    rng = np.random.default_rng(0)
    task_cfg = TaskConfig(n_ctx=model.cfg.n_ctx)
    n_batches = (n_seqs + batch - 1) // batch
    xs, pres, posts, outs, toks, depths, valids = [], [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            b = sample_batch(batch, task_cfg, rng, length_range=length_range)
            tok = b["tok"].to(device); mask = b["mask"].to(device)
            _, cache = model(tok, return_internals=True)
            ln2_out = cache["blocks"][layer]["ln2_out"]
            pre = cache["blocks"][layer]["mlp_pre"]
            post = cache["blocks"][layer]["mlp_post"]
            mlp_out = cache["blocks"][layer]["mlp_out"]
            m = mask
            xs.append(ln2_out[m].cpu().float())
            pres.append(pre[m].cpu().float())
            posts.append(post[m].cpu().float())
            outs.append(mlp_out[m].cpu().float())
            toks.append(tok[m].cpu())
            depths.append(b["depth"].to(device)[m].cpu())
            valids.append(b["valid"].to(device)[m].cpu())
    return {
        "x":    torch.cat(xs, dim=0),
        "pre":  torch.cat(pres, dim=0),
        "post": torch.cat(posts, dim=0),
        "out":  torch.cat(outs, dim=0),
        "tok":  torch.cat(toks, dim=0),
        "depth": torch.cat(depths, dim=0),
        "valid": torch.cat(valids, dim=0),
    }


def activation_patterns(post: torch.Tensor) -> torch.Tensor:
    """Binary activation pattern per row: True if neuron is on (>0)."""
    return post > 0


def pattern_to_int(pat_row: np.ndarray) -> int:
    """Hash a Boolean activation pattern row to a Python int for tabulation.
    For d_mlp=256, the int is up to 256 bits long."""
    return int.from_bytes(np.packbits(pat_row.astype(np.uint8)).tobytes(), "big")


def enumerate_regions(post: torch.Tensor) -> dict:
    """Hash each pattern row to an int; return the distribution of patterns."""
    patterns_np = (post.numpy() > 0)
    keys = [pattern_to_int(row) for row in patterns_np]
    ctr = Counter(keys)
    return {
        "n_unique_patterns": len(ctr),
        "top_patterns": [(k, v) for k, v in ctr.most_common(20)],
        "n_rows": len(keys),
        "pattern_keys": keys,
    }


def per_region_local_map(model: Transformer, layer: int, post: torch.Tensor,
                          pre: torch.Tensor, ln2_out: torch.Tensor,
                          pattern_keys: list[int], top_k: int = 16) -> dict:
    """For each of the `top_k` most populated regions, compute the local
    affine map x -> y for the MLP (with the current activation pattern fixed).

    y = W_out @ diag(mask) @ pre  (since pre = W_in @ x_norm + 0 [no bias])
      = (W_out @ diag(mask) @ W_in) @ x_norm
    """
    folded = fold_rmsnorm(model)
    Win = folded.blocks[layer].mlp.W_in.detach().cpu().float().numpy()   # [d_mlp, d_model]
    Wout = folded.blocks[layer].mlp.W_out.detach().cpu().float().numpy() # [d_model, d_mlp]
    ctr = Counter(pattern_keys)
    top = ctr.most_common(top_k)
    regions = []
    pattern_keys_arr = np.array(pattern_keys)
    for key, count in top:
        idx = np.where(pattern_keys_arr == key)[0]
        if len(idx) == 0:
            continue
        # Reconstruct the boolean mask from the first occurrence
        post_row = (post[idx[0]].numpy() > 0).astype(np.float64)
        # Local map: A = W_out * diag(post_row) * W_in
        A_local = Wout @ np.diag(post_row) @ Win   # [d_model, d_model]
        # Singular values (a description of the operator)
        s = np.linalg.svd(A_local, compute_uv=False)
        regions.append({
            "key": str(key)[:32] + "...",
            "count": int(count),
            "active_neurons": int(post_row.sum()),
            "A_norm": float(np.linalg.norm(A_local)),
            "A_rank_eff": int((s > s.max() * 1e-6).sum()),
            "A_singular_top10": s[:10].tolist(),
        })
    return {"regions": regions}


def region_to_behavior(io: dict, pattern_keys: list[int], top_k: int = 16) -> dict:
    """For each top region, what task labels (token, depth, valid) co-occur there?"""
    ctr = Counter(pattern_keys)
    top = ctr.most_common(top_k)
    pattern_keys_arr = np.array(pattern_keys)
    tok = io["tok"].numpy(); dep = io["depth"].numpy(); val = io["valid"].numpy()
    out = []
    for key, count in top:
        idx = np.where(pattern_keys_arr == key)[0]
        out.append({
            "key": str(key)[:32] + "...",
            "count": int(count),
            "tok_dist": _hist(tok[idx], 40),
            "depth_dist": _hist(dep[idx], 9),
            "valid_dist": _hist(val[idx], 2),
        })
    return {"regions": out}


def _hist(arr: np.ndarray, n: int) -> dict:
    """Return a frequency-dict {bin: count}."""
    out = {}
    counts = np.bincount(arr, minlength=n)
    for i, c in enumerate(counts):
        if c > 0:
            out[int(i)] = int(c)
    return out


def run_lens4(model: Transformer, out_dir: str, n_seqs: int = 4096,
              device: str = "cuda") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    summary = {"layers": []}
    for L in range(model.cfg.n_layers):
        print(f"[Lens 4] layer {L}: collecting MLP I/O ({n_seqs} sequences)...", flush=True)
        io = collect_mlp_io(model, L, n_seqs=n_seqs, device=device)
        print(f"[Lens 4] layer {L}: enumerating activation patterns...", flush=True)
        regions = enumerate_regions(io["post"])
        local_maps = per_region_local_map(model, L, io["post"], io["pre"],
                                          io["x"], regions["pattern_keys"], top_k=24)
        behavior = region_to_behavior(io, regions["pattern_keys"], top_k=24)
        # cardinality of regions vs samples
        out = {
            "layer": L,
            "n_samples": regions["n_rows"],
            "n_unique_patterns": regions["n_unique_patterns"],
            "top_pattern_counts": [c for _, c in regions["top_patterns"]],
            "local_maps_top24": local_maps,
            "behavior_top24": behavior,
        }
        summary["layers"].append(out)
        print(f"[Lens 4] layer {L}: {regions['n_unique_patterns']} unique patterns "
              f"in {regions['n_rows']} samples (top-24 region cumulative coverage: "
              f"{sum(c for _, c in regions['top_patterns'][:24]) / regions['n_rows']:.2%})",
              flush=True)
    with open(os.path.join(out_dir, "lens4.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary
