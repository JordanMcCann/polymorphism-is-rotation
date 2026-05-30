"""Experiment 1a: SAE reconstruction transfer.

Question: when seed 0's SAE is applied to seed N's residual stream
activations on identical inputs, how well does it reconstruct?

Measurement: explained variance and reconstruction MSE per (site, seed,
expansion). Controls:
  * within-seed (seed 0 SAE on seed 0 activations) -- ceiling
  * shuffled-position activations -- floor
  * Gaussian noise matched to seed-N covariance -- noise baseline

We additionally include the *bf16-noise floor*: re-applying seed 0's SAE
to seed 0's activations from a different sampling seed (the SAE was trained
on a different sample of the activation distribution).
"""

from __future__ import annotations

import json
import os

import torch

from .utils import (
    SITES,
    collect_acts_flat,
    load_sae,
    load_seed_model,
    make_eval_batch,
)

SEEDS = [0, 1, 2, 3, 4]


@torch.no_grad()
def sae_metrics(sae, acts: torch.Tensor) -> dict:
    """Reconstruction MSE, explained variance, L0 sparsity, fraction-firing.

    acts: [N, d_in] flattened activations.
    """
    recon, feats = sae(acts)
    err = recon - acts
    recon_mse = float((err ** 2).mean().item())
    var_x = float(acts.var().item())
    explained_var = 1.0 - recon_mse / max(var_x, 1e-12)
    sparsity_l0 = float((feats > 0).float().sum(dim=1).mean().item())
    feature_freq = (feats > 0).float().mean(dim=0)  # [d_feat]
    fraction_active = float((feature_freq > 0).float().mean().item())
    return {
        "recon_mse": recon_mse,
        "var_x": var_x,
        "explained_var": explained_var,
        "sparsity_l0": sparsity_l0,
        "fraction_features_active": fraction_active,
        "mean_feature_freq": float(feature_freq.mean().item()),
        "n_samples": int(acts.shape[0]),
    }


@torch.no_grad()
def shuffled_baseline(sae, acts: torch.Tensor, seed: int = 7) -> dict:
    """Apply SAE to a per-dimension permutation of the activation matrix.

    Destroys cross-dimension correlation while preserving the marginal of
    each residual dim. A useful floor: it tells us how well the SAE would
    reconstruct activations that lie 'in distribution per coordinate' but
    not on the trained manifold.
    """
    g = torch.Generator(device=acts.device).manual_seed(seed)
    N, d = acts.shape
    shuffled = torch.empty_like(acts)
    for j in range(d):
        idx = torch.randperm(N, generator=g, device=acts.device)
        shuffled[:, j] = acts[idx, j]
    return sae_metrics(sae, shuffled)


@torch.no_grad()
def gaussian_baseline(sae, acts: torch.Tensor, seed: int = 7) -> dict:
    """SAE on Gaussian noise matched to mean+covariance of acts.

    Uses eigendecomposition (robust to rank-deficient covariance, which is
    typical for resid_pre_0 -- only a handful of W_pos/W_E dims are active)."""
    mean = acts.mean(dim=0)
    cov = torch.cov(acts.T)
    # Eigendecomp; clip tiny negative eigvals from numerical noise
    evals, evecs = torch.linalg.eigh(cov.double())
    evals = evals.clamp(min=0)
    L = evecs * evals.sqrt().unsqueeze(0)            # [d, d]
    g = torch.Generator(device=acts.device).manual_seed(seed)
    z = torch.randn(acts.shape[0], acts.shape[1], generator=g, device=acts.device).double()
    samples = (mean.double().unsqueeze(0) + z @ L.T).float()
    return sae_metrics(sae, samples)


def run(device: str = "cuda", batch_size: int = 1024,
         expansions: tuple[int, ...] = (8, 32),
         out_path: str = "experiments/cross_seed/exp1a_reconstruction.json"):
    print(f"[exp1a] loading seeds 0..{max(SEEDS)} and SAEs from seed 0", flush=True)
    models = {s: load_seed_model(s, device=device)[0] for s in SEEDS}
    # Single fixed eval batch — identical inputs to every seed
    batch = make_eval_batch(batch_size=batch_size, length_range=(2, 48))
    print(f"[exp1a] batch: {batch['tok'].shape}, valid tokens: "
          f"{int(batch['mask'].sum().item())}", flush=True)

    # Collect activations from each seed at every site
    print("[exp1a] collecting activations from each seed", flush=True)
    acts_by_seed = {}
    for s in SEEDS:
        acts_by_seed[s] = collect_acts_flat(models[s], SITES, batch, device=device)

    # Run each seed-0 SAE on every seed's activations
    results = {}
    for site in SITES:
        results[site] = {}
        for exp in expansions:
            try:
                sae = load_sae(0, site, exp, device=device)
            except FileNotFoundError:
                print(f"  [skip] no SAE for site={site} x{exp}", flush=True)
                continue
            results[site][f"x{exp}"] = {}
            for s in SEEDS:
                m = sae_metrics(sae, acts_by_seed[s][site].float())
                results[site][f"x{exp}"][f"seed{s}"] = m
            # Controls (use seed 0's activations as the reference distribution)
            results[site][f"x{exp}"]["control_shuffled_seed0"] = \
                shuffled_baseline(sae, acts_by_seed[0][site].float())
            results[site][f"x{exp}"]["control_gaussian_seed0"] = \
                gaussian_baseline(sae, acts_by_seed[0][site].float())
            # And a second-sample baseline: a fresh batch on seed 0
            sec_batch = make_eval_batch(batch_size=batch_size, length_range=(2, 48),
                                         seed=99999)
            sec_acts = collect_acts_flat(models[0], [site], sec_batch, device=device)[site].float()
            results[site][f"x{exp}"]["control_secondsample_seed0"] = sae_metrics(sae, sec_acts)

            ev0 = results[site][f"x{exp}"]["seed0"]["explained_var"]
            evs = [results[site][f"x{exp}"][f"seed{s}"]["explained_var"]
                   for s in SEEDS if s != 0]
            ev_min, ev_max = min(evs), max(evs)
            sh = results[site][f"x{exp}"]["control_shuffled_seed0"]["explained_var"]
            print(f"  site={site:<14} x{exp}: EV(seed0)={ev0:.4f}  "
                  f"EV(seed1..4)=[{ev_min:.4f}, {ev_max:.4f}]  "
                  f"EV(shuffled)={sh:.4f}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp1a] wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--expansions", type=int, nargs="+", default=[8, 32])
    p.add_argument("--out", default="experiments/cross_seed/exp1a_reconstruction.json")
    args = p.parse_args()
    run(device=args.device, batch_size=args.batch_size,
        expansions=tuple(args.expansions), out_path=args.out)
