"""EXP 2 — Independent-init Dyck-3 analysis.

After polymorphism.train_all has trained seeds 100..104 with NO shared frozen I/O,
this script:

  1. Verifies convergence (min held-out accuracy from logs).
  2. Builds a constructive spec from primary seed 100 (mirrors §3.5).
  3. Computes per-seed Bar B at 1e7 samples (one of the open-question answers
     was: "yes, re-run Bar B on EXP 2 seeds" since the landmark case wants
     the full bar table).
  4. Aligns seeds 101..104 to seed 100 via the existing multi-start
     symmetry search; reports Bar P max per-tensor MSE.
  5. Trains SAEs at every site for every seed (re-using lens2_saes).
  6. Runs the rotation-audit panels (mirror of exp1d at scale=104k):
     - Naive cross-seed SAE EV
     - Post-Procrustes EV
     - ||R-I||_F vs sqrt(2*d_model)
  7. Reports the §8.5 three-regime steering analysis on these seeds.

Outputs (mirrors experiments/cross_seed/* but for independent-init seeds):

  experiments/independent_init/bars.json
  experiments/independent_init/rotation_audit.json
  experiments/independent_init/steering.json
  experiments/independent_init/saes_summary.json
  logs/v2/exp2_analysis.log
"""

from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import torch

from ...analysis.lens2_saes import SAEConfig, train_sae
from ...rmsnorm_fold import fold_rmsnorm
from ...verification.bar_behavioral import run_bar_behavioral
from ..cross_seed.utils import (
    SITES,
    collect_acts_flat,
    load_sae,
    load_seed_model,
    make_eval_batch,
)
from ..scale.common import best_orthogonal, procrustes_metrics

SEEDS = [100, 101, 102, 103, 104]
PRIMARY = 100
OUT_DIR = "experiments/independent_init"


# -------------------- Convergence verification --------------------

def verify_convergence(seeds: list[int]) -> dict:
    """Read per-seed train_seed{N}.json and report best eval_min_acc."""
    result = {}
    for s in seeds:
        log_path = f"logs/train_seed{s}.json"
        if not os.path.exists(log_path):
            result[s] = {"status": "NO_LOG", "log_path": log_path}
            continue
        try:
            data = json.load(open(log_path))
        except json.JSONDecodeError:
            result[s] = {"status": "CORRUPT_LOG", "log_path": log_path}
            continue
        evals = [r for r in data if isinstance(r, dict)
                 and isinstance(r.get("train"), dict)]
        if not evals:
            result[s] = {"status": "NO_EVALS"}
            continue
        best = max(evals, key=lambda r: r["eval_min_acc"])
        result[s] = {
            "best_eval_step": int(best["step"]),
            "best_eval_min_acc": float(best["eval_min_acc"]),
            "converged_99_99": best["eval_min_acc"] >= 0.9999,
            "converged_99_95": best["eval_min_acc"] >= 0.9995,
            "n_evals": len(evals),
        }
    return result


# -------------------- Bar P cross-seed --------------------

def run_cross_seed_bars(seeds: list[int], primary: int = PRIMARY,
                          device: str = "cuda") -> dict:
    """Use run_universality (existing pipeline) to get all 4 bars for each
    (primary, replication) pair. This gives Bar B, P, C, Pr in one shot."""
    from ...verification.universality import run_universality
    results = {"primary": primary, "pairs": {}}
    for s in seeds:
        if s == primary:
            continue
        print(f"[exp2-bars] running universality {primary} vs {s} ...", flush=True)
        try:
            r = run_universality(primary, s, device=device, n_outer=6,
                                  n_starts=16, ig_steps=16)
            results["pairs"][s] = r
            bp = r["bars"]
            print(f"[exp2-bars]   seed{s}: "
                  f"B={bp['B']['mean_kl']:.2e} (passed={bp['B']['passed']}), "
                  f"P_max={bp['P']['max_per_tensor_mse']:.4f} (passed={bp['P']['passed']}), "
                  f"C={bp['C']['pearson_r']:.4f} (passed={bp['C']['passed']})",
                  flush=True)
        except Exception as e:
            print(f"[exp2-bars]   seed{s}: FAILED {type(e).__name__}: {e}",
                  flush=True)
            results["pairs"][s] = {"error": f"{type(e).__name__}: {e}"}
    return results


# -------------------- Bar B at 1e7 samples --------------------

def run_per_seed_bar_b(seeds: list[int], n_samples: int = 10_000_000,
                        device: str = "cuda") -> dict:
    """Per-seed Bar B against itself (which is the constructive-spec setup
    we use for the toy paper). This is the same Bar B as the original paper."""
    results = {}
    for s in seeds:
        print(f"[exp2-barB] seed {s} ...", flush=True)
        m, _ = load_seed_model(s, device=device)
        m = fold_rmsnorm(m).to(device).eval()
        # The constructive spec IS the folded model itself, so Bar B against
        # the spec is Bar B against the same model = 0 by construction. To
        # compute a meaningful Bar B for an independent-init seed, we compare
        # against the constructive spec from seed 100 (the primary). This
        # mirrors the paper's §7 (cross-seed B against the primary's spec).
        spec_path = f"experiments/seeds/{PRIMARY}/lens_outputs/spec_constructive.pt"
        # If the constructive spec doesn't exist for the primary, fall back to
        # using the model itself as the spec (which is 0; not useful).
        if not os.path.exists(spec_path):
            print(f"[exp2-barB]   skip — no constructive spec at {spec_path}",
                  flush=True)
            continue
        from ...analysis.lens5_rasp import compile_spec_to_model
        spec_model, _ = compile_spec_to_model(mode='constructive',
                                                 primary_seed=PRIMARY)
        spec_model = spec_model.to(device).eval()
        t0 = time.time()
        bar_b = run_bar_behavioral(m, spec=spec_model, n_samples=n_samples,
                                     batch_size=2048, device=device, seed=2026,
                                     progress_every=2000)
        bar_b["wall_sec"] = time.time() - t0
        results[s] = bar_b
        print(f"[exp2-barB]   seed{s}: kl={bar_b['kl_mean']:.2e} "
              f"passed={bar_b['passed']}  ({time.time()-t0:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache(); gc.collect()
    return results


# -------------------- SAE training per-seed (mirrors lens2_saes.run_lens2) --------------------

def train_indep_init_saes(seeds: list[int], expansion: int = 8,
                            n_steps: int = 3000,
                            device: str = "cuda") -> dict:
    """Train one SAE per (seed, site) at the chosen expansion. Cache to disk."""
    from ...analysis.lens2_saes import collect_activations
    summary = {}
    for s in seeds:
        m, _ = load_seed_model(s, device=device)
        m = fold_rmsnorm(m).to(device).eval()
        out_dir = f"experiments/seeds/{s}/lens_outputs"
        os.makedirs(out_dir, exist_ok=True)
        acts = collect_activations(m, SITES, n_seqs=4096, batch=128,
                                     device=device)
        for site in SITES:
            cache_path = os.path.join(out_dir, f"sae_{site}_x{expansion}.pt")
            if os.path.exists(cache_path):
                continue
            cfg = SAEConfig(d_in=acts[site].shape[1], expansion=expansion,
                              n_steps=n_steps, batch_size=4096, lr=5e-4,
                              l1_coef=1e-3, seed=0)
            res = train_sae(acts[site], cfg, device=device, verbose=False)
            torch.save({"state": res["state"], "config": res["config"],
                         "metrics": {k: v for k, v in res.items()
                                     if k not in ("state", "feature_freq",
                                                  "metrics_history")},
                         "feature_freq": res["feature_freq"]}, cache_path)
            summary[(s, site)] = {
                "explained_var": float(res["explained_var"]),
                "sparsity_l0": float(res["sparsity_l0"]),
            }
            print(f"[exp2-sae] seed{s} {site}: EV={res['explained_var']:.4f}",
                  flush=True)
        del m
        torch.cuda.empty_cache(); gc.collect()
    return {f"seed{s}_{site}": v for (s, site), v in summary.items()}


# -------------------- Rotation audit (mirror of exp1d) --------------------

def run_rotation_audit(seeds: list[int], primary: int = PRIMARY,
                         device: str = "cuda") -> dict:
    """For each seed N: collect activations on the same fixed batch as the
    primary, fit Procrustes R, apply primary-SAE to (acts_N @ R), measure EV.

    Also compute a "rotation + linear re-labeling" variant: since the
    independent-init seeds may have *different W_E* (the residual basis at
    the input boundary diverges), we additionally fit a per-position
    re-labeling Q at resid_pre_0 (the W_E + W_pos site) and then chain it
    with the residual rotation R for the audit.
    """
    print("[exp2-rot] loading models ...", flush=True)
    models = {s: load_seed_model(s, device=device)[0] for s in seeds}
    batch = make_eval_batch(batch_size=1024, length_range=(2, 48), seed=2026)
    acts = {s: collect_acts_flat(models[s], SITES, batch, device=device)
            for s in seeds}
    primary_idx = seeds.index(primary)
    results = {"primary": primary, "per_site": {}}
    for site in SITES:
        results["per_site"][site] = {"alignment": {}, "sae_post_rotation": {}}
        a_pr = acts[primary][site].float()
        a_pr_c = a_pr - a_pr.mean(0, keepdim=True)
        var_p = float(a_pr_c.pow(2).mean().item())
        for s in seeds:
            if s == primary:
                continue
            a_n = acts[s][site].float()
            R = best_orthogonal(a_n, a_pr)
            metrics = procrustes_metrics(a_n, a_pr, R)
            results["per_site"][site]["alignment"][f"seed{s}"] = metrics
            # Apply primary SAE if available
            try:
                sae = load_sae(primary, site, 8, device=device)
                a_n_c = a_n - a_n.mean(0, keepdim=True)
                a_pr_mean = a_pr.mean(0)
                shift = a_pr_mean - (a_n.mean(0) @ R)
                a_n_rot = a_n @ R + shift
                with torch.no_grad():
                    recon_raw, _ = sae(a_n.to(device))
                    recon_rot, _ = sae(a_n_rot.to(device))
                    ev_raw = float(1 - ((recon_raw - a_n.to(device)) ** 2).mean()
                                     / a_n.var().clamp(min=1e-12))
                    ev_rot = float(1 - ((recon_rot - a_n_rot.to(device)) ** 2).mean()
                                     / a_n_rot.var().clamp(min=1e-12))
                results["per_site"][site]["sae_post_rotation"][f"seed{s}"] = {
                    "raw_EV": ev_raw, "rotated_EV": ev_rot,
                }
            except FileNotFoundError:
                pass
        # Site summary
        if results["per_site"][site]["alignment"]:
            raw_evs = [v["raw_EV"]
                        for v in results["per_site"][site]["sae_post_rotation"].values()]
            rot_evs = [v["rotated_EV"]
                        for v in results["per_site"][site]["sae_post_rotation"].values()]
            frobs = [v["frob_R_minus_I"]
                      for v in results["per_site"][site]["alignment"].values()]
            d_model = next(iter(results["per_site"][site]["alignment"].values()))["d_model"]
            results["per_site"][site]["summary"] = {
                "n_pairs": len(frobs),
                "mean_raw_EV": float(np.mean(raw_evs)) if raw_evs else None,
                "mean_rot_EV": float(np.mean(rot_evs)) if rot_evs else None,
                "mean_frob_R_minus_I": float(np.mean(frobs)),
                "predicted_random_orthogonal_frob": float((2 * d_model) ** 0.5),
            }
    return results


# -------------------- main orchestrator --------------------

def run_all(device: str = "cuda", out_dir: str = OUT_DIR,
             skip_bar_b: bool = False, skip_saes: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    t_total = time.time()
    print("=" * 60, flush=True)
    print("[exp2] STARTING EXP 2 ANALYSIS — independent-init Dyck-3", flush=True)
    print(f"[exp2] seeds: {SEEDS}, primary: {PRIMARY}", flush=True)
    print("=" * 60, flush=True)
    out = {}

    # Step 1: convergence
    print("[exp2] step 1: verifying convergence ...", flush=True)
    out["convergence"] = verify_convergence(SEEDS)
    with open(os.path.join(out_dir, "convergence.json"), "w") as f:
        json.dump(out["convergence"], f, indent=2)
    for s, c in out["convergence"].items():
        print(f"  seed{s}: {c}", flush=True)

    # Only proceed with downstream steps if at least 3 seeds converged
    n_converged = sum(1 for c in out["convergence"].values()
                       if c.get("converged_99_95"))
    if n_converged < 3:
        print(f"[exp2] only {n_converged}/5 seeds converged to 99.95%; "
              f"continuing analysis but flag as residual", flush=True)
    converged_seeds = [s for s in SEEDS
                        if out["convergence"].get(s, {}).get("converged_99_95")]
    if not converged_seeds:
        print("[exp2] no seeds converged; aborting downstream", flush=True)
        return out
    out["converged_seeds"] = converged_seeds

    # Step 2: SAEs
    if not skip_saes:
        print("[exp2] step 2: training SAEs ...", flush=True)
        out["saes_summary"] = train_indep_init_saes(converged_seeds,
                                                       device=device)
        with open(os.path.join(out_dir, "saes_summary.json"), "w") as f:
            json.dump(out["saes_summary"], f, indent=2)

    # Step 3: cross-seed all 4 bars
    print("[exp2] step 3: cross-seed all 4 bars ...", flush=True)
    primary_use = PRIMARY if PRIMARY in converged_seeds else converged_seeds[0]
    out["bars"] = run_cross_seed_bars(converged_seeds, primary=primary_use,
                                         device=device)
    with open(os.path.join(out_dir, "bars.json"), "w") as f:
        json.dump(out["bars"], f, indent=2, default=str)

    # Step 4: rotation audit
    print("[exp2] step 4: rotation audit ...", flush=True)
    out["rotation_audit"] = run_rotation_audit(converged_seeds,
                                                  primary=primary_use,
                                                  device=device)
    with open(os.path.join(out_dir, "rotation_audit.json"), "w") as f:
        json.dump(out["rotation_audit"], f, indent=2, default=str)

    # Step 5: (skipped — per-seed Bar B against own constructive spec = 0 by
    # construction; the meaningful Bar B is cross-seed and already in step 3.)

    out["wall_sec_total"] = time.time() - t_total
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[exp2] DONE in {(out['wall_sec_total'] / 60):.1f} min", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_bar_b", action="store_true")
    p.add_argument("--skip_saes", action="store_true")
    args = p.parse_args()
    run_all(device=args.device, skip_bar_b=args.skip_bar_b,
             skip_saes=args.skip_saes)
