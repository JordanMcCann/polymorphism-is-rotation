"""Experiment 1b: SAE firing-pattern overlap.

On identical inputs, encode seed-0 residual stream and seed-N residual stream
through the SAME (seed-0-trained) SAE. For each feature f, ask:
   * Pearson correlation across (sequence, position) of seed-0 firing vs
     seed-N firing.
   * Jaccard similarity of the top-K firing positions.
   * Mean firing rate ratio (seed-N / seed-0).

We also include a "rotated" variant: apply the best orthogonal R (Procrustes
on the activation matrices) before encoding seed-N. This separates
"coordinate-frame rotation" from "true firing-pattern change".

Controls:
   * Within-seed (seed 0 SAE on seed 0's same batch -- the trivial 1.0 line)
   * Within-seed-different-batch (seed 0 vs seed 0 with two fresh samplings,
     so the matrix shape matches but content varies -- captures sampling
     noise rather than seed-difference effect; here we compute it on a
     per-FEATURE level, where the comparison is on the marginal firing rate).
   * Cross-feature shuffled (random feature permutation in seed N -- floor).
"""

from __future__ import annotations

import json
import os

import torch

from .utils import (
    collect_acts_flat,
    load_sae,
    load_seed_model,
    make_eval_batch,
)

SEEDS = [0, 1, 2, 3, 4]


@torch.no_grad()
def best_orthogonal(acts_src: torch.Tensor, acts_tgt: torch.Tensor) -> torch.Tensor:
    """Procrustes: orthogonal R minimising ||acts_src @ R - acts_tgt||_F.

    Returns R [d, d]."""
    a_src = (acts_src - acts_src.mean(0, keepdim=True)).double()
    a_tgt = (acts_tgt - acts_tgt.mean(0, keepdim=True)).double()
    M = a_src.T @ a_tgt
    U, _, Vt = torch.linalg.svd(M)
    return (U @ Vt).to(acts_src.dtype)


@torch.no_grad()
def feature_overlap(feats_a: torch.Tensor, feats_b: torch.Tensor,
                     top_k: int = 100) -> dict:
    """Compare two feature activation matrices [N, d_feat].

    Per feature:
      * Pearson correlation of firing magnitudes across the N samples.
      * Jaccard similarity of top-K firing positions.
      * Firing-rate match (fraction of features whose marginal active rate
        in B is within 10% of A's).
    """
    N, F = feats_a.shape
    # Pearson per feature; vectorised
    a_c = feats_a - feats_a.mean(dim=0, keepdim=True)
    b_c = feats_b - feats_b.mean(dim=0, keepdim=True)
    a_n = a_c.norm(dim=0)
    b_n = b_c.norm(dim=0)
    denom = (a_n * b_n).clamp(min=1e-12)
    pearson = (a_c * b_c).sum(dim=0) / denom            # [F]

    # Top-K position Jaccard per feature (only features with at least K
    # nonzero firings in both)
    active_a = (feats_a > 0)
    active_b = (feats_b > 0)
    fire_count_a = active_a.float().sum(dim=0)
    fire_count_b = active_b.float().sum(dim=0)

    # For each feature, find the top-K (by activation magnitude) positions in A and B
    K = min(top_k, N)
    topk_a = feats_a.topk(K, dim=0).indices              # [K, F]
    topk_b = feats_b.topk(K, dim=0).indices              # [K, F]
    # Build masks
    mask_a = torch.zeros(N, F, dtype=torch.bool, device=feats_a.device)
    mask_b = torch.zeros_like(mask_a)
    mask_a.scatter_(0, topk_a, True)
    mask_b.scatter_(0, topk_b, True)
    # Restrict to features that fire >= K times in both (else Jaccard is degenerate)
    valid_feats = (fire_count_a >= K) & (fire_count_b >= K)
    inter = (mask_a & mask_b).float().sum(dim=0)
    union = (mask_a | mask_b).float().sum(dim=0).clamp(min=1)
    jaccard = inter / union                              # [F]

    # Firing-rate match
    rate_a = active_a.float().mean(dim=0)
    rate_b = active_b.float().mean(dim=0)
    rate_diff_abs = (rate_a - rate_b).abs()

    valid_pearson = ~torch.isnan(pearson)
    return {
        "n_features": int(F),
        "n_samples": int(N),
        "mean_pearson_all": float(pearson[valid_pearson].mean().item()),
        "median_pearson_all": float(pearson[valid_pearson].median().item()),
        "mean_pearson_active": float(
            pearson[valid_pearson & (rate_a > 0) & (rate_b > 0)].mean().item()
        ) if ((valid_pearson & (rate_a > 0) & (rate_b > 0)).any()) else float("nan"),
        "fraction_pearson_above_0p5": float((pearson[valid_pearson] > 0.5).float().mean().item()),
        "fraction_pearson_above_0p9": float((pearson[valid_pearson] > 0.9).float().mean().item()),
        "mean_jaccard_top_k": float(jaccard[valid_feats].mean().item())
            if valid_feats.any() else float("nan"),
        "fraction_jaccard_above_0p5": float((jaccard[valid_feats] > 0.5).float().mean().item())
            if valid_feats.any() else float("nan"),
        "mean_firing_rate_a": float(rate_a.mean().item()),
        "mean_firing_rate_b": float(rate_b.mean().item()),
        "mean_firing_rate_diff_abs": float(rate_diff_abs.mean().item()),
        "n_features_active_in_both": int(((rate_a > 0) & (rate_b > 0)).sum().item()),
        "top_k": int(K),
        "n_features_valid_for_jaccard": int(valid_feats.sum().item()),
    }


def run(device: str = "cuda", batch_size: int = 1024,
         sites: tuple[str, ...] = ("resid_mid_0", "resid_post_0", "resid_pre_1",
                                     "resid_post_1", "resid_pre_2"),
         expansions: tuple[int, ...] = (8, 32),
         out_path: str = "experiments/cross_seed/exp1b_firing_overlap.json"):
    print("[exp1b] loading models, building activations", flush=True)
    models = {s: load_seed_model(s, device=device)[0] for s in SEEDS}
    batch = make_eval_batch(batch_size=batch_size, length_range=(2, 48))
    # Second batch for the within-seed "ceiling" baseline
    batch2 = make_eval_batch(batch_size=batch_size, length_range=(2, 48),
                              seed=999999)
    acts = {s: collect_acts_flat(models[s], list(sites), batch, device=device)
            for s in SEEDS}
    acts_seed0_b2 = collect_acts_flat(models[0], list(sites), batch2, device=device)

    results = {}
    for site in sites:
        results[site] = {}
        for exp in expansions:
            try:
                sae = load_sae(0, site, exp, device=device)
            except FileNotFoundError:
                continue
            # Reference: encode seed 0 acts on batch 1
            a0 = acts[0][site].float()
            feats0 = sae.encode(a0)
            # Cross-seed: encode seed N acts on the SAME batch with the
            # SAME SAE
            per_seed = {}
            for s in SEEDS:
                a = acts[s][site].float()
                feats = sae.encode(a)
                per_seed[f"seed{s}_raw"] = feature_overlap(feats0, feats)
                # Rotation-corrected: align seed N acts to seed 0 acts first
                R = best_orthogonal(a, a0)
                rot_a = a @ R
                feats_rot = sae.encode(rot_a)
                per_seed[f"seed{s}_rotated"] = feature_overlap(feats0, feats_rot)
            # Within-seed ceiling: different batch on seed 0
            a0_b2 = acts_seed0_b2[site].float()
            feats0_b2 = sae.encode(a0_b2)
            # Since shapes differ across batches, only marginals are comparable
            per_seed["control_seed0_diffbatch_marginal"] = {
                "mean_firing_rate_a": float((feats0 > 0).float().mean().item()),
                "mean_firing_rate_b": float((feats0_b2 > 0).float().mean().item()),
                "abs_diff": float(((feats0 > 0).float().mean(dim=0)
                                    - (feats0_b2 > 0).float().mean(dim=0)
                                    ).abs().mean().item()),
            }
            results[site][f"x{exp}"] = per_seed
            r0 = per_seed["seed0_raw"]
            r1 = per_seed["seed1_raw"]; r1r = per_seed["seed1_rotated"]
            print(f"  site={site:<14} x{exp}: "
                  f"seed0 self pearson={r0['mean_pearson_all']:.3f}  "
                  f"seed1 raw={r1['mean_pearson_all']:.3f} "
                  f"rotated={r1r['mean_pearson_all']:.3f}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp1b] wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--out", default="experiments/cross_seed/exp1b_firing_overlap.json")
    args = p.parse_args()
    run(device=args.device, batch_size=args.batch_size, out_path=args.out)
