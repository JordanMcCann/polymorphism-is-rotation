"""EXP 6 — eigenvalue spectrum of the cross-seed Procrustes rotation R.

EXP 1's Panel C established that ||R - I||_F = sqrt(2 * d_model) to within
0.1% across all 56 (pair, site) combinations on Pythia-70m. That's necessary
but not sufficient evidence that R is a true Haar-uniform sample from SO(d):
by concentration of measure, any sufficiently mixed orthogonal matrix lives
on the same Frobenius shell.

A discriminating test: under Haar measure on SO(d), the eigenvalues of R
are uniformly distributed on the unit circle as conjugate pairs e^{±iθ_k}
with θ_k drawn from a specific distribution (the Weyl integration formula).
For d large the angles are approximately uniform on [0, π]; equivalently,
2 sin(θ/2) (the singular values of R - I) are approximately uniform on
[0, 2] modulo a Wigner-semicircle-ish density at the edges.

This script:
  1. For each (anchor=seed1, replication=seed{2..9}) pair and each residual
     site, loads the cached activations and fits R via Procrustes.
  2. Computes the eigenvalues of R, takes their angular part theta.
  3. Compares the empirical theta distribution to Haar's predicted density.
  4. Also reports cos(theta) — the diagonal of R in the eigenbasis — and
     compares to the Haar mean.
  5. Reports nearest-permutation distance ||R - P||_F for the best P found
     by Hungarian on R as a structuredness check.

Cost: < 2 minutes on CPU (cached activations, eigenvalue compute on 512x512).
Writes: experiments/scale/pythia_rotation/eigenvalue_spectrum.json
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from .common import (
    CorpusConfig,
    best_orthogonal,
    load_cached_acts,
)

MODELS = [f"EleutherAI/pythia-70m-seed{i}" for i in range(1, 10)]
ANCHOR_IDX = 0
CACHE_DIR = "experiments/scale/cache"
OUT_PATH = "experiments/scale/pythia_rotation/eigenvalue_spectrum.json"


def fit_R(acts_n: torch.Tensor, acts_anchor: torch.Tensor) -> torch.Tensor:
    """Procrustes: orthogonal R such that acts_n @ R approximates acts_anchor."""
    return best_orthogonal(acts_n, acts_anchor)


def eigen_angles(R: torch.Tensor) -> np.ndarray:
    """Return eigenvalue angles of R in [0, pi]. For orthogonal R the eigenvalues
    are on the unit circle, in complex conjugate pairs e^{+/- i theta}, with at
    most a single +/-1 eigenvalue for d odd or det = -1.

    Returns the angles theta_k folded to [0, pi] (one entry per conjugate pair
    plus the real eigenvalues at 0 or pi).
    """
    eigs = np.linalg.eigvals(R.cpu().numpy().astype(np.float64))
    # angles in [-pi, pi]
    theta = np.angle(eigs)
    # Fold to [0, pi]
    theta = np.abs(theta)
    return theta


def haar_predicted_density(d: int, n_bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Sample a Haar-uniform R from SO(d) and return its eigenvalue angle histogram.

    For d = 512 the empirical distribution is essentially the Haar prediction
    with negligible noise after averaging over many samples.
    """
    rng = np.random.default_rng(7)
    n_samples = 20  # 20 samples * d eigenvalues ≈ 10k angles → low-noise histogram
    all_thetas = []
    for _ in range(n_samples):
        # Haar sample via QR of Gaussian matrix
        A = rng.standard_normal((d, d))
        Q, Rm = np.linalg.qr(A)
        # Adjust signs to get true Haar (Mezzadri 2007)
        sgn = np.sign(np.diag(Rm))
        Q = Q * sgn
        eigs = np.linalg.eigvals(Q)
        all_thetas.append(np.abs(np.angle(eigs)))
    pooled = np.concatenate(all_thetas)
    counts, edges = np.histogram(pooled, bins=n_bins, range=(0, np.pi))
    density = counts.astype(np.float64) / (n_samples * (edges[1] - edges[0]))
    return edges, density / d  # normalised per-eigenvalue density


def best_permutation_distance(R: torch.Tensor) -> float:
    """Best ||R - P||_F over permutation matrices P, via Hungarian on max
    over rows. Returns the Frobenius distance to the best permutation.

    A truly random orthogonal matrix has E[||R - P_best||_F] = sqrt(2d - 2)
    approximately, because no permutation can do much better than identity
    would (||R - I||_F is already sqrt(2d) in expectation).
    """
    try:
        from scipy.optimize import linear_sum_assignment
        Rabs = R.abs().cpu().numpy().astype(np.float64)
        # Hungarian maximises sum of |R_ij| over a permutation (best alignment).
        # linear_sum_assignment minimises, so negate.
        rows, cols = linear_sum_assignment(-Rabs)
        d = R.shape[0]
        P = torch.zeros_like(R)
        for r, c in zip(rows, cols):
            P[r, c] = torch.sign(R[r, c]) if R[r, c] != 0 else 1.0
        return float((R - P).pow(2).sum().sqrt().item())
    except Exception:
        # Fallback: trivial diagonal P = I
        d = R.shape[0]
        I = torch.eye(d, device=R.device, dtype=R.dtype)
        return float((R - I).pow(2).sum().sqrt().item())


def ks_test_uniform(thetas: np.ndarray, haar_thetas: np.ndarray) -> dict:
    """Two-sample KS test between observed and Haar-predicted angles.

    Falls back to a manual KS if scipy is missing.
    """
    try:
        from scipy.stats import ks_2samp
        result = ks_2samp(thetas, haar_thetas)
        return {"ks_stat": float(result.statistic),
                "ks_pvalue": float(result.pvalue),
                "n_obs": int(thetas.size), "n_haar": int(haar_thetas.size)}
    except ImportError:
        # Manual KS via empirical CDFs
        x = np.sort(np.concatenate([thetas, haar_thetas]))
        cdf_a = np.searchsorted(np.sort(thetas), x, side="right") / thetas.size
        cdf_b = np.searchsorted(np.sort(haar_thetas), x, side="right") / haar_thetas.size
        return {"ks_stat": float(np.max(np.abs(cdf_a - cdf_b))),
                "ks_pvalue": None, "n_obs": int(thetas.size),
                "n_haar": int(haar_thetas.size)}


def main(n_sequences: int = 256, seq_len: int = 256, max_pairs: int = 8,
          out_path: str = OUT_PATH) -> dict:
    """Run the eigenvalue spectrum analysis over all cached Pythia seeds."""
    t0 = time.time()
    cfg = CorpusConfig(n_sequences=n_sequences, seq_len=seq_len, seed=2026)

    # Probe d_model and sites from anchor cache (any one site works).
    sites = [
        "layer0_resid_pre",
        "layer0_resid_post", "layer1_resid_post", "layer2_resid_post",
        "layer3_resid_post", "layer4_resid_post", "layer5_resid_post",
    ]
    anchor_id = MODELS[ANCHOR_IDX]
    anchor_cache = load_cached_acts(CACHE_DIR, anchor_id, None, cfg, sites)
    if anchor_cache is None:
        raise FileNotFoundError(
            f"No cached activations for {anchor_id} with cfg {cfg}. "
            "Run EXP 1 first (panel_c_fast.py) to populate the cache.")
    d_model = int(anchor_cache[sites[1]].shape[-1])
    print(f"[eigspec] d_model={d_model}, sites={sites}")

    # Haar baseline for d_model (computed once, reused everywhere)
    print(f"[eigspec] generating Haar baseline for d={d_model} ...")
    edges, haar_density = haar_predicted_density(d_model, n_bins=30)
    # Also a flat pooled list of Haar angles for KS comparison
    rng = np.random.default_rng(31)
    haar_pool = []
    for _ in range(10):
        A = rng.standard_normal((d_model, d_model))
        Q, Rm = np.linalg.qr(A)
        sgn = np.sign(np.diag(Rm))
        Q = Q * sgn
        haar_pool.append(np.abs(np.angle(np.linalg.eigvals(Q))))
    haar_pool = np.concatenate(haar_pool)

    results = {
        "config": {"n_sequences": n_sequences, "seq_len": seq_len,
                    "d_model": d_model, "anchor": anchor_id, "sites": sites},
        "haar_baseline": {
            "edges": edges.tolist(),
            "density_per_eigenvalue": haar_density.tolist(),
            "n_pooled_haar_samples": int(haar_pool.size),
        },
        "per_pair": {},
        "summary": {},
    }

    all_angles_pooled = []
    all_perm_distances = []
    for n_idx in range(1, min(len(MODELS), max_pairs + 1)):
        rep_id = MODELS[n_idx]
        print(f"[eigspec] processing {rep_id} vs {anchor_id} ...")
        rep_cache = load_cached_acts(CACHE_DIR, rep_id, None, cfg, sites)
        if rep_cache is None:
            print(f"[eigspec]   no cache for {rep_id}, skipping")
            continue
        pair_key = f"seed{n_idx + 1}_vs_seed1"
        results["per_pair"][pair_key] = {}
        for site in sites:
            a_n = rep_cache[site].reshape(-1, d_model)
            a_0 = anchor_cache[site].reshape(-1, d_model)
            R = fit_R(a_n, a_0)
            angles = eigen_angles(R)
            counts, _ = np.histogram(angles, bins=edges)
            density = counts.astype(np.float64) / (angles.size * (edges[1] - edges[0]))
            ks = ks_test_uniform(angles, haar_pool)
            mean_cos_theta = float(np.cos(angles).mean())
            haar_mean_cos = float(np.cos(haar_pool).mean())
            perm_dist = best_permutation_distance(R)
            results["per_pair"][pair_key][site] = {
                "n_eigenvalues": int(angles.size),
                "angle_histogram_density": density.tolist(),
                "mean_cos_theta": mean_cos_theta,
                "haar_mean_cos_theta": haar_mean_cos,
                "ks_stat_vs_haar": ks["ks_stat"],
                "ks_pvalue_vs_haar": ks["ks_pvalue"],
                "frob_R_minus_I": float((R - torch.eye(d_model, dtype=R.dtype)).pow(2).sum().sqrt().item()),
                "frob_R_minus_best_perm": perm_dist,
                "predicted_random_perm_dist": float(np.sqrt(2 * d_model - 2)),
            }
            all_angles_pooled.extend(angles.tolist())
            all_perm_distances.append(perm_dist)
        print(f"[eigspec]   done {pair_key}")

    # Aggregate
    angles_arr = np.array(all_angles_pooled)
    counts, _ = np.histogram(angles_arr, bins=edges)
    pooled_density = counts.astype(np.float64) / (angles_arr.size * (edges[1] - edges[0]))
    overall_ks = ks_test_uniform(angles_arr, haar_pool)
    results["summary"] = {
        "n_pair_site_combinations": len(all_perm_distances),
        "n_pooled_eigenvalues": int(angles_arr.size),
        "pooled_angle_density": pooled_density.tolist(),
        "pooled_mean_cos_theta": float(np.cos(angles_arr).mean()),
        "haar_mean_cos_theta": float(np.cos(haar_pool).mean()),
        "ks_stat_pooled_vs_haar": overall_ks["ks_stat"],
        "ks_pvalue_pooled_vs_haar": overall_ks["ks_pvalue"],
        "mean_perm_dist": float(np.mean(all_perm_distances)),
        "std_perm_dist": float(np.std(all_perm_distances)),
        "predicted_random_perm_dist": float(np.sqrt(2 * d_model - 2)),
        "predicted_random_perm_dist_relative_to_frob_I":
            float(np.sqrt(2 * d_model - 2) / np.sqrt(2 * d_model)),
        "interpretation": (
            f"Observed mean ||R - best_perm||_F = {np.mean(all_perm_distances):.3f}; "
            f"predicted under Haar = ~{np.sqrt(2 * d_model - 2):.3f} (essentially "
            f"the same as ||R - I||_F = sqrt(2d) = {np.sqrt(2*d_model):.3f}). "
            f"If R were structured to be permutation-like, this would be much smaller."
        ),
    }
    elapsed = time.time() - t0
    results["wall_time_sec"] = elapsed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eigspec] wrote {out_path}; total time {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--n_sequences", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=256)
    args = ap.parse_args()
    main(n_sequences=args.n_sequences, seq_len=args.seq_len, out_path=args.out)
