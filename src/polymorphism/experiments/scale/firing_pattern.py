"""EXP 7 — Pythia-70m firing-pattern overlap (raw vs post-rotation).

The toy-scale §8.5 paragraph established that after orthogonal rotation, per-
feature firing-pattern Pearson correlation between seed-0 and seed-N lifts
from [0.03, 0.51] (raw) to [0.27, 0.74] (rotated). The reconstruction
recovers fully (0.97-0.99 EV) but firing patterns recover only partially —
suggesting residual non-orthogonal structure on top of the dominant rotation.

This script reproduces that measurement at Pythia-70m scale, closing the
"Pythia firing-pattern not measured" limitation in PAPER §10.

Method (per anchor=seed1, replication=seed{2..9}, site):
  1. Load seed1's trained SAE (x8 expansion) and the cached activations
     for both seeds.
  2. Encode both seed1 and seed-N activations through seed1's SAE; get per-
     position feature activations [N_pos, d_feat].
  3. For each feature column k, compute Pearson(seed1_feat[k], seedN_feat[k]).
     Average over features. This is the "raw" firing pattern correlation.
  4. Fit Procrustes R between seedN and seed1 activations; encode the
     rotated seedN activations through seed1's SAE; recompute correlations.
     This is the "post-rotation" firing pattern correlation.
  5. Report mean/median/quartiles of the per-feature correlations.

Cost: ~5-10 minutes on RTX 2060 (encode is fast; the 56 (pair, site)
combinations are independent and small).
Writes: experiments/scale/pythia_rotation/firing_pattern.json
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
    sae_apply,
)

MODELS = [f"EleutherAI/pythia-70m-seed{i}" for i in range(1, 10)]
ANCHOR_IDX = 0
SAE_DIR = "experiments/scale/cache/saes"
CACHE_DIR = "experiments/scale/cache"
OUT_PATH = "experiments/scale/pythia_rotation/firing_pattern.json"
SITES = [
    "layer0_resid_pre",
    "layer0_resid_post", "layer1_resid_post", "layer2_resid_post",
    "layer3_resid_post", "layer4_resid_post", "layer5_resid_post",
]


def safe_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def load_sae(model_id: str, site: str, expansion: int = 8) -> tuple[dict, dict] | None:
    path = os.path.join(SAE_DIR,
                          f"sae_{safe_id(model_id)}_{site}_x{expansion}.pt")
    if not os.path.exists(path):
        return None
    st = torch.load(path, map_location="cpu", weights_only=False)
    return st["state"], st["config"]


def per_feature_pearson(feat_a: torch.Tensor, feat_b: torch.Tensor,
                          eps: float = 1e-8) -> np.ndarray:
    """Per-feature Pearson r between two activation matrices [N, d_feat].

    Returns a numpy array of length d_feat. Features with zero variance in
    either matrix get NaN, filtered out in summary stats by the caller.
    """
    a = feat_a.numpy().astype(np.float64)
    b = feat_b.numpy().astype(np.float64)
    a_c = a - a.mean(0, keepdims=True)
    b_c = b - b.mean(0, keepdims=True)
    num = (a_c * b_c).sum(0)
    denom = np.sqrt((a_c ** 2).sum(0) * (b_c ** 2).sum(0) + eps)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.where(denom == 0, np.nan, denom)
    return r


def summarise(r: np.ndarray) -> dict:
    valid = r[~np.isnan(r)]
    if valid.size == 0:
        return {"mean": float("nan"), "median": float("nan"),
                "n_features": 0, "n_valid_features": 0}
    return {
        "mean": float(valid.mean()),
        "median": float(np.median(valid)),
        "p05": float(np.percentile(valid, 5)),
        "p25": float(np.percentile(valid, 25)),
        "p75": float(np.percentile(valid, 75)),
        "p95": float(np.percentile(valid, 95)),
        "n_features": int(r.size),
        "n_valid_features": int(valid.size),
        "fraction_above_0p5": float((valid > 0.5).mean()),
        "fraction_above_0p9": float((valid > 0.9).mean()),
    }


def main(n_sequences: int = 256, seq_len: int = 256, device: str = "cuda",
          expansion: int = 8, n_acts_for_R: int = 8192,
          n_acts_for_corr: int = 32768,
          out_path: str = OUT_PATH) -> dict:
    """Compute Pythia firing-pattern overlap, raw vs post-rotation."""
    t0 = time.time()
    cfg = CorpusConfig(n_sequences=n_sequences, seq_len=seq_len, seed=2026)

    anchor_id = MODELS[ANCHOR_IDX]
    anchor_acts_full = load_cached_acts(CACHE_DIR, anchor_id, None, cfg, SITES)
    if anchor_acts_full is None:
        raise FileNotFoundError(f"No cached activations for {anchor_id}")
    d_model = int(anchor_acts_full[SITES[1]].shape[-1])
    print(f"[firing_pattern] d_model={d_model}, sites={len(SITES)}")

    # Load anchor SAEs
    anchor_saes = {}
    for s in SITES:
        sae = load_sae(anchor_id, s, expansion)
        if sae is None:
            print(f"[firing_pattern]  no SAE for {anchor_id} {s}; skipping site")
            continue
        anchor_saes[s] = sae
    if not anchor_saes:
        raise RuntimeError("No anchor SAEs found at expansion x8")

    results = {
        "config": {"n_sequences": n_sequences, "seq_len": seq_len,
                    "expansion": expansion, "d_model": d_model,
                    "anchor": anchor_id,
                    "n_acts_for_R": n_acts_for_R,
                    "n_acts_for_corr": n_acts_for_corr},
        "per_pair_per_site": {},
        "summary_per_site": {},
    }

    # Flatten anchor activations once
    anchor_flat = {s: a.reshape(-1, d_model) for s, a in anchor_acts_full.items()
                    if s in anchor_saes}

    # Cap to n_acts_for_corr
    rng = np.random.default_rng(2027)
    sub_idx_corr = {}
    for s, a in anchor_flat.items():
        n = a.shape[0]
        take = min(n_acts_for_corr, n)
        idx = rng.choice(n, take, replace=False)
        sub_idx_corr[s] = idx
        anchor_flat[s] = a[idx]

    # Encode anchor activations once
    print("[firing_pattern] encoding anchor activations through anchor SAEs ...")
    anchor_feats = {}
    for s in anchor_saes:
        state, sae_cfg = anchor_saes[s]
        _, feats = sae_apply(state, sae_cfg, anchor_flat[s], device=device)
        anchor_feats[s] = feats
        print(f"[firing_pattern]   {s}: feats shape {tuple(feats.shape)}")

    # Per replication
    for n_idx in range(1, len(MODELS)):
        rep_id = MODELS[n_idx]
        print(f"[firing_pattern] {rep_id} ...")
        rep_acts_full = load_cached_acts(CACHE_DIR, rep_id, None, cfg, SITES)
        if rep_acts_full is None:
            print(f"[firing_pattern]  no cache for {rep_id}, skipping")
            continue
        pair_key = f"seed{n_idx + 1}_vs_seed1"
        results["per_pair_per_site"][pair_key] = {}

        for s in anchor_saes:
            rep_full = rep_acts_full[s].reshape(-1, d_model)
            # Use the same sub-indices for direct alignment
            rep_sub = rep_full[sub_idx_corr[s]]
            state, sae_cfg = anchor_saes[s]
            # Encode raw replication acts through anchor SAE
            _, rep_feats_raw = sae_apply(state, sae_cfg, rep_sub, device=device)
            # Fit R on first n_acts_for_R activations (separate from corr set for honesty)
            n_for_R = min(n_acts_for_R, rep_full.shape[0])
            R_fit_idx = rng.choice(rep_full.shape[0], n_for_R, replace=False)
            rep_for_R = rep_full[R_fit_idx]
            anchor_for_R = anchor_acts_full[s].reshape(-1, d_model)[R_fit_idx]
            R = best_orthogonal(rep_for_R, anchor_for_R)
            # Recentre + rotate the corr set
            rep_centred = rep_sub - rep_full.mean(0, keepdim=True)
            anchor_mean = anchor_acts_full[s].reshape(-1, d_model).mean(0, keepdim=True)
            rep_rotated = rep_centred @ R + anchor_mean
            _, rep_feats_rot = sae_apply(state, sae_cfg, rep_rotated, device=device)
            # Per-feature Pearson
            r_raw = per_feature_pearson(anchor_feats[s], rep_feats_raw)
            r_rot = per_feature_pearson(anchor_feats[s], rep_feats_rot)
            results["per_pair_per_site"][pair_key][s] = {
                "raw": summarise(r_raw),
                "rotated": summarise(r_rot),
            }
        print(f"[firing_pattern]   {pair_key} done")

    # Summary per site (mean across pairs)
    for s in anchor_saes:
        raw_means = []
        rot_means = []
        raw_frac_above_p5 = []
        rot_frac_above_p5 = []
        for pair_key, pair_dat in results["per_pair_per_site"].items():
            if s in pair_dat:
                raw_means.append(pair_dat[s]["raw"]["mean"])
                rot_means.append(pair_dat[s]["rotated"]["mean"])
                raw_frac_above_p5.append(pair_dat[s]["raw"]["fraction_above_0p5"])
                rot_frac_above_p5.append(pair_dat[s]["rotated"]["fraction_above_0p5"])
        results["summary_per_site"][s] = {
            "mean_raw_pearson": float(np.mean(raw_means)) if raw_means else float("nan"),
            "mean_rotated_pearson": float(np.mean(rot_means)) if rot_means else float("nan"),
            "mean_raw_fraction_above_0p5": float(np.mean(raw_frac_above_p5)) if raw_frac_above_p5 else float("nan"),
            "mean_rotated_fraction_above_0p5": float(np.mean(rot_frac_above_p5)) if rot_frac_above_p5 else float("nan"),
            "n_pairs": len(raw_means),
        }

    elapsed = time.time() - t0
    results["wall_time_sec"] = elapsed
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[firing_pattern] wrote {out_path}; total time {elapsed:.1f}s")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--n_sequences", type=int, default=256)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_acts_for_corr", type=int, default=32768)
    ap.add_argument("--n_acts_for_R", type=int, default=8192)
    args = ap.parse_args()
    main(n_sequences=args.n_sequences, seq_len=args.seq_len,
          device=args.device, n_acts_for_corr=args.n_acts_for_corr,
          n_acts_for_R=args.n_acts_for_R, out_path=args.out)
