"""Experiment 1d (extension): residual-stream rotation audit.

Result we observed in exp1a and exp1b: cross-seed SAE encoder/decoder
operations fail dramatically (EV goes from ~0.999 within-seed to negative
across seeds) DESPITE behavioural KL being ~10^-5. The discrepancy is
mostly explained by an orthogonal rotation of the residual basis.

This script measures, per site, three numbers:
   raw_EV          : explained variance of identity map seed_N -> seed_0
   rot_EV          : best orthogonal-rotation EV
   norm_R_minus_I  : how far R is from identity

It also reports a "rotation-induced lower bound on SAE recon error":
applying seed-0 SAE to (best_R @ seed_N_acts) instead of seed_N_acts.
If the SAE still does well post-rotation, the SAE features are
preserved up to rotation -- a stronger universality result than raw
decoder cosine similarity.
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
def best_orthogonal(acts_src: torch.Tensor, acts_tgt: torch.Tensor) -> torch.Tensor:
    a_src = (acts_src - acts_src.mean(0, keepdim=True)).double()
    a_tgt = (acts_tgt - acts_tgt.mean(0, keepdim=True)).double()
    M = a_src.T @ a_tgt
    U, _, Vt = torch.linalg.svd(M)
    return (U @ Vt).to(acts_src.dtype)


@torch.no_grad()
def sae_metrics(sae, acts: torch.Tensor) -> dict:
    recon, feats = sae(acts)
    err = recon - acts
    recon_mse = float((err ** 2).mean().item())
    var_x = float(acts.var().item())
    explained_var = 1.0 - recon_mse / max(var_x, 1e-12)
    return {"recon_mse": recon_mse, "explained_var": explained_var,
            "sparsity_l0": float((feats > 0).float().sum(dim=1).mean().item()),
            "n": int(acts.shape[0])}


def run(device: str = "cuda", batch_size: int = 1024,
         expansions: tuple[int, ...] = (8, 32),
         out_path: str = "experiments/cross_seed/exp1d_rotation_audit.json"):
    print("[exp1d] loading models", flush=True)
    models = {s: load_seed_model(s, device=device)[0] for s in SEEDS}
    batch = make_eval_batch(batch_size=batch_size, length_range=(2, 48))
    acts = {s: collect_acts_flat(models[s], SITES, batch, device=device)
            for s in SEEDS}

    results = {}
    for site in SITES:
        results[site] = {"alignment": {}, "sae_post_rotation": {}}
        a0 = acts[0][site].float()
        a0c = a0 - a0.mean(0, keepdim=True)
        var_a0 = float(a0c.pow(2).mean().item())
        for s in SEEDS:
            if s == 0:
                continue
            a_n = acts[s][site].float()
            a_nc = a_n - a_n.mean(0, keepdim=True)
            R = best_orthogonal(a_n, a0)
            rotated = a_nc @ R
            err_raw = (a_nc - a0c).pow(2).mean().item()
            err_rot = (rotated - a0c).pow(2).mean().item()
            ev_raw = 1.0 - err_raw / max(var_a0, 1e-12)
            ev_rot = 1.0 - err_rot / max(var_a0, 1e-12)
            I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
            results[site]["alignment"][f"seed{s}"] = {
                "raw_EV_vs_seed0":      float(ev_raw),
                "rot_EV_vs_seed0":      float(ev_rot),
                "frob_R_minus_I":       float((R - I).pow(2).sum().sqrt().item()),
                "op_norm_R":            float(torch.linalg.norm(R, ord=2).item()),
            }
            # Apply seed-0 SAE to rotated seed-N activations
            results[site]["sae_post_rotation"].setdefault(f"seed{s}", {})
            for exp in expansions:
                try:
                    sae = load_sae(0, site, exp, device=device)
                except FileNotFoundError:
                    continue
                # raw
                m_raw = sae_metrics(sae, a_n)
                m_rot = sae_metrics(sae, a_n @ R + (a0.mean(0) - (a_n.mean(0) @ R)))
                results[site]["sae_post_rotation"][f"seed{s}"][f"x{exp}_raw"] = m_raw
                results[site]["sae_post_rotation"][f"seed{s}"][f"x{exp}_rotated"] = m_rot

        # Concise log
        if results[site]["alignment"]:
            s1 = results[site]["alignment"].get("seed1", {})
            raw1 = s1.get("raw_EV_vs_seed0", float("nan"))
            rot1 = s1.get("rot_EV_vs_seed0", float("nan"))
            sae_raw = results[site]["sae_post_rotation"].get("seed1", {}).get("x8_raw", {}).get("explained_var", float("nan"))
            sae_rot = results[site]["sae_post_rotation"].get("seed1", {}).get("x8_rotated", {}).get("explained_var", float("nan"))
            print(f"  {site:<14} seed1 vs seed0: raw_EV={raw1:.3f} rot_EV={rot1:.3f} | "
                  f"SAE x8 raw_EV={sae_raw:.3f} post-rot EV={sae_rot:.3f}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp1d] wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--out", default="experiments/cross_seed/exp1d_rotation_audit.json")
    args = p.parse_args()
    run(device=args.device, batch_size=args.batch_size, out_path=args.out)
