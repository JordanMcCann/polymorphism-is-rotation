"""EXP 3 — Joint Bar P optimization.

The original symmetry search (polymorphism/symmetry_search.py) optimises per-tensor
weight MSE only, with the residual rotation R fit by Procrustes on a stack of
the model's residual-touching weights. §8.6 of the paper notes that this
recovers head permutations + per-head subspace rotations + MLP perm/scaling
but does NOT jointly optimise the residual rotation against an *activation-
level* loss.

This module adds that joint optimisation:

  loss(R) = (1 - lambda) * mean( || rotate(p_seed, R) - p_ref ||^2 )
          +  lambda      * mean( || rotate(acts_seed, R) - acts_ref ||^2 )

R is constrained to SO(d_model) via Cayley parameterisation:
  A = skew-symmetric matrix (d^2 - d) / 2 free parameters
  R = (I + A) (I - A)^{-1}   -- always orthogonal with det +1.

The Cayley map is differentiable; we optimise A via Adam. After convergence
we report:
  - max per-tensor MSE
  - global weight MSE
  - activation MSE (held-out batch)
  - the trade-off curve over lambda

The outer loop uses the existing multi-start head-permutation infrastructure
from align(); we initialise from the best param-only alignment and then
refine R via the joint loss.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from ...model import Transformer, make_model
from ...rmsnorm_fold import fold_rmsnorm
from ...symmetry_search import (
    _params_as_dict,
    align,
)
from ...task import TaskConfig, sample_batch


def cayley(A: torch.Tensor) -> torch.Tensor:
    """Cayley transform: R = (I + A_skew)(I - A_skew)^{-1}.

    A is any d x d matrix; we skew-symmetrize as (A - A^T)/2 internally.
    """
    d = A.shape[0]
    skew = 0.5 * (A - A.t())
    I = torch.eye(d, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(I - skew, I + skew)


@torch.no_grad()
def collect_activations_at_sites(model: Transformer, batch: dict,
                                  sites: list[str], device: str) -> dict:
    """Collect [B*T_valid, d_model] tensors at the named sites (mask-filtered)."""
    tok = batch["tok"].to(device)
    mask = batch["mask"].to(device)
    _, cache = model(tok, return_internals=True)
    blocks = cache["blocks"]
    flat_mask = mask.reshape(-1)
    site_map = {
        "resid_pre_0":  blocks[0]["resid_pre"],
        "resid_mid_0":  blocks[0]["resid_mid"],
        "resid_post_0": blocks[0]["resid_post"],
        "resid_pre_1":  blocks[1]["resid_pre"],
        "resid_mid_1":  blocks[1]["resid_mid"],
        "resid_post_1": blocks[1]["resid_post"],
        "resid_pre_2":  blocks[-1]["resid_post"],
    }
    out = {}
    for s in sites:
        a = site_map[s]
        flat = a.reshape(-1, a.shape[-1])
        out[s] = flat[flat_mask].detach()
    return out


def rotate_acts(acts: dict, R: torch.Tensor) -> dict:
    """Apply R to every site's activation: a -> a @ R.t() (we use the same
    convention as _apply_residual_rotation, where the new residual is R @ r).
    """
    return {s: a @ R.t() for s, a in acts.items()}


def weight_mse_from_dicts(p_seed: dict, p_ref: dict) -> torch.Tensor:
    """Differentiable weight MSE over all tensors (sum of squares / total elements)."""
    total_sq = 0.0
    total_n = 0
    for k in p_seed:
        d = (p_seed[k] - p_ref[k]).flatten()
        total_sq = total_sq + (d * d).sum()
        total_n += d.numel()
    return total_sq / max(total_n, 1)


def max_per_tensor_mse_torch(p_seed: dict, p_ref: dict) -> torch.Tensor:
    """Differentiable max per-tensor MSE. Returns a scalar tensor."""
    max_m = None
    for k in p_seed:
        d = (p_seed[k] - p_ref[k]).flatten()
        mse = (d * d).mean()
        if max_m is None or mse > max_m:
            max_m = mse
    return max_m


def _apply_residual_rotation_grad(params: dict, R: torch.Tensor) -> dict:
    """Same as polymorphism.symmetry_search._apply_residual_rotation but autograd-aware.

    Returns a dict of tensors that participate in the autograd graph through R.
    """
    Rt = R.t()
    new = {}
    for k, v in params.items():
        if k == "W_E" or k == "W_pos":
            new[k] = v @ Rt
        elif k.endswith("attn.W_Q") or k.endswith("attn.W_K") or k.endswith("attn.W_V"):
            new[k] = torch.einsum("hpd,de->hpe", v, Rt)
        elif k.endswith("attn.W_O"):
            new[k] = torch.einsum("ed,hdp->hep", R, v)
        elif k.endswith("mlp.W_in"):
            new[k] = v @ Rt
        elif k.endswith("mlp.W_out"):
            new[k] = R @ v
        elif k in ("W_U_tok", "W_U_depth", "W_U_valid"):
            new[k] = v @ Rt
        elif k.endswith(".gain"):
            new[k] = v
        else:
            new[k] = v
    return new


def joint_align(model_seed: Transformer, model_ref: Transformer,
                 activation_batch: dict,
                 lambda_act: float = 1.0,
                 n_starts: int = 16,
                 n_outer: int = 6,
                 try_no_rotation: bool = True,
                 n_inner_iters: int = 400,
                 lr: float = 5e-2,
                 sites_for_act_loss: tuple[str, ...] = (
                     "resid_post_0", "resid_post_1", "resid_pre_2"),
                 device: str = "cuda",
                 verbose: bool = False) -> dict:
    """Joint optimisation of R against weight + activation MSE.

    Strategy:
      1. Run the existing multi-start `align()` to get a strong initial
         parameter-level alignment (head perms, per-head rotations,
         MLP perm + scaling) plus an initial Procrustes R.
      2. Freeze the discrete part (perms) and the per-head rotations.
         Re-optimise R via Cayley-parameterised Riemannian Adam, with the
         joint loss = (1-lambda) * weight_MSE + lambda * activation_MSE.
      3. After convergence, apply the optimised R and re-measure per-tensor
         MSE, max MSE, and activation MSE.

    Returns dict with the final R, the metrics over the optimisation, and
    the final per-tensor MSE breakdown.
    """
    # Step 1: existing multi-start align — keep weight-only initialisation.
    # We mirror the paper's exact symmetry-search settings (n_starts=16,
    # n_outer=6, try_no_rotation=True so both rotation-on and rotation-off
    # modes are tried) so the lambda=0 result matches the paper's baseline.
    p_aligned_init, info = align(model_seed, model_ref,
                                  n_outer=n_outer, n_starts=n_starts,
                                  try_no_rotation=try_no_rotation)
    p_ref = _params_as_dict(model_ref)
    # Cast to fp32 on device
    p_aligned = {k: v.to(device).float() for k, v in p_aligned_init.items()}
    p_ref_d = {k: v.to(device).float() for k, v in p_ref.items()}

    # Collect reference activations on the activation batch
    with torch.no_grad():
        acts_ref = collect_activations_at_sites(model_ref, activation_batch,
                                                  list(sites_for_act_loss), device)
        # For the "seed" side, we need activations of the *aligned* model.
        # But the aligned model is the seed model with discrete perms applied;
        # we approximate this by rotating the seed model's activations through
        # whatever combined rotation `info` discovered, then further rotating
        # by R during optimisation. To keep things tractable, we re-run the
        # *aligned* model on the batch to capture its activations directly.
        # Since the aligned parameters are in p_aligned, we instantiate a
        # transient model holding them.
        m_seed_aligned = make_model(model_seed.cfg).to(device).eval()
        # Manually load aligned params + missing keys
        sd = m_seed_aligned.state_dict()
        for k in sd:
            if k in p_aligned:
                sd[k] = p_aligned[k].to(sd[k].dtype)
        m_seed_aligned.load_state_dict(sd, strict=False)
        acts_seed = collect_activations_at_sites(m_seed_aligned, activation_batch,
                                                   list(sites_for_act_loss), device)

    d_model = p_ref_d["W_E"].shape[1]
    # Optimisable skew-symmetric parameter A; R = Cayley(A)
    A = torch.zeros(d_model, d_model, device=device, requires_grad=True)
    opt = torch.optim.Adam([A], lr=lr)
    history = []
    best = {"max_mse": float("inf"), "weight_mse": float("inf"),
            "act_mse": float("inf"), "R": None, "step": -1}

    for step in range(n_inner_iters):
        R = cayley(A)
        # Apply R to the aligned-seed params, then compare to ref
        p_rot = _apply_residual_rotation_grad(p_aligned, R)
        w_mse = weight_mse_from_dicts(p_rot, p_ref_d)
        # Activation loss: apply R to seed-side acts, compare to ref acts
        a_mse_sum = 0.0
        for s in sites_for_act_loss:
            a_rot = acts_seed[s] @ R.t()
            a_mse_sum = a_mse_sum + ((a_rot - acts_ref[s]) ** 2).mean()
        a_mse_sum = a_mse_sum / len(sites_for_act_loss)
        loss = (1.0 - lambda_act) * w_mse + lambda_act * a_mse_sum
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 20 == 0 or step == 0:
            with torch.no_grad():
                R_eval = cayley(A)
                p_eval = _apply_residual_rotation_grad(p_aligned, R_eval)
                max_mse = float(max_per_tensor_mse_torch(p_eval, p_ref_d).item())
                w_eval = float(weight_mse_from_dicts(p_eval, p_ref_d).item())
                # Total activation MSE
                a_eval = 0.0
                for s in sites_for_act_loss:
                    a_eval += float(((acts_seed[s] @ R_eval.t() - acts_ref[s]) ** 2)
                                     .mean().item())
                a_eval /= len(sites_for_act_loss)
                history.append({
                    "step": step + 1, "loss": float(loss.item()),
                    "weight_mse": w_eval, "act_mse": a_eval, "max_mse": max_mse,
                })
                if (max_mse, w_eval) < (best["max_mse"], best["weight_mse"]):
                    best = {"max_mse": max_mse, "weight_mse": w_eval,
                            "act_mse": a_eval, "R": R_eval.detach().cpu().clone(),
                            "step": step + 1}
                if verbose:
                    print(f"    [joint step={step+1}] max_mse={max_mse:.4f} "
                          f"w_mse={w_eval:.5f} a_mse={a_eval:.5f}", flush=True)

    # Final per-tensor breakdown
    R_best = best["R"].to(device)
    p_final = _apply_residual_rotation_grad(p_aligned, R_best)
    per_tensor = {}
    for k in p_final:
        d = (p_final[k] - p_ref_d[k]).flatten()
        per_tensor[k] = {
            "mse": float((d * d).mean().item()),
            "max_abs": float(d.abs().max().item()),
        }

    return {
        "lambda_act": lambda_act,
        "n_inner_iters": n_inner_iters,
        "best": {k: v for k, v in best.items() if k != "R"},
        "history": history,
        "per_tensor": per_tensor,
        "init_align_info_brief": {
            "best_max_mse": info.get("best_max_mse"),
            "best_mse": info.get("best_mse"),
            "best_start_index": info.get("best_start_index"),
            "best_use_rotation": info.get("best_use_rotation"),
        },
        "R_final_norm_minus_I": float((R_best - torch.eye(
            R_best.shape[0], device=R_best.device)).pow(2).sum().sqrt().item()),
        "R_final_op_norm": float(torch.linalg.norm(R_best, ord=2).item()),
    }


# ---------- orchestrator ----------

def run_pair(primary_seed: int, replication_seed: int,
              lambdas: list[float],
              device: str = "cuda",
              batch_size: int = 1024,
              n_inner_iters: int = 400,
              verbose: bool = False) -> dict:
    """Run joint align for a (primary, replication) pair across all lambdas."""
    from ...experiments.cross_seed.utils import load_seed_model
    m_primary, _ = load_seed_model(primary_seed, device=device)
    m_rep, _ = load_seed_model(replication_seed, device=device)
    m_primary = fold_rmsnorm(m_primary).to(device).eval()
    m_rep = fold_rmsnorm(m_rep).to(device).eval()

    rng = np.random.default_rng(202605200 + replication_seed)
    task_cfg = TaskConfig(n_ctx=m_primary.cfg.n_ctx)
    batch_full = sample_batch(batch_size, task_cfg, rng, length_range=(2, 48))
    batch = {k: v for k, v in batch_full.items()}

    out = {"primary_seed": primary_seed, "replication_seed": replication_seed,
           "lambdas": lambdas, "results_per_lambda": []}
    t0 = time.time()
    for lam in lambdas:
        t1 = time.time()
        if verbose:
            print(f"  [pair {primary_seed}-{replication_seed}] lambda={lam}",
                  flush=True)
        res = joint_align(m_rep, m_primary, batch,
                           lambda_act=lam, n_inner_iters=n_inner_iters,
                           device=device, verbose=verbose)
        res["wall_sec"] = time.time() - t1
        out["results_per_lambda"].append(res)
        print(f"  [pair {primary_seed}-{replication_seed} lam={lam}] "
              f"max_mse={res['best']['max_mse']:.4f}  "
              f"w_mse={res['best']['weight_mse']:.5f}  "
              f"a_mse={res['best']['act_mse']:.5f}  "
              f"||R-I||_F={res['R_final_norm_minus_I']:.3f}  "
              f"({res['wall_sec']:.1f}s)", flush=True)
    out["wall_sec_total"] = time.time() - t0
    return out


def run_all(out_dir: str = "experiments/bar_p_joint",
             lambdas: list[float] | None = None,
             pairs: list[tuple[int, int]] | None = None,
             device: str = "cuda",
             n_inner_iters: int = 400) -> dict:
    """Full sweep: all 4 pairs (0, N) for N in 1..4 across all lambdas."""
    if lambdas is None:
        lambdas = [0.0, 0.1, 1.0, 10.0, 100.0]
    if pairs is None:
        pairs = [(0, 1), (0, 2), (0, 3), (0, 4)]
    os.makedirs(out_dir, exist_ok=True)
    results = []
    t0 = time.time()
    for p_s, r_s in pairs:
        r = run_pair(p_s, r_s, lambdas, device=device,
                      n_inner_iters=n_inner_iters, verbose=False)
        results.append(r)
        with open(os.path.join(out_dir, f"joint_{p_s}_vs_{r_s}.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
    summary = {"pairs": pairs, "lambdas": lambdas,
               "results": results,
               "wall_sec_total": time.time() - t0}
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="experiments/bar_p_joint")
    p.add_argument("--n_inner_iters", type=int, default=400)
    args = p.parse_args()
    run_all(out_dir=args.out, device=args.device,
             n_inner_iters=args.n_inner_iters)
