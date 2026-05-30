"""Phase 6 (Universality): align each replication seed to the primary
and verify all 4 bars under that alignment.

This script:
  1. Loads the primary seed (seed 0) and a replication seed (k>0).
  2. Folds RMSNorm in both.
  3. Searches for the symmetry-group element that aligns seed_k to seed_0.
  4. Applies the alignment to seed_k's weights to produce aligned_k.
  5. Runs all four bars comparing aligned_k to seed_0:
     - B: KL(seed_0 model output ‖ aligned_k model output)
     - P: per-entry weight MSE between seed_0 and aligned_k
     - C: Pearson r of patch effects measured on each model
     - Pr: Pearson r of ablation losses measured on each model

If any of the four bars fail for a given seed, that is recorded as a
universality break -- a discovered finding to be reported.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from polymorphism.analysis.lens3_causal import component_ablation_table, mean_activations
    from polymorphism.model import Config, Transformer, make_model
    from polymorphism.rmsnorm_fold import fold_rmsnorm
    from polymorphism.symmetry_search import _param_mse, _params_as_dict, align
    from polymorphism.task import TaskConfig, sample_batch
else:
    from ..analysis.lens3_causal import component_ablation_table, mean_activations
    from ..model import Config, Transformer, make_model
    from ..rmsnorm_fold import fold_rmsnorm
    from ..symmetry_search import _param_mse, _params_as_dict, align
    from ..task import TaskConfig, sample_batch


def load_seed(seed: int, device: str = "cpu", which: str = "best"):
    """Load a checkpoint. which='best' picks the highest min_acc with lowest
    total loss as tiebreaker; 'last' picks final."""
    ckpts = sorted(glob.glob(f"experiments/seeds/{seed}/checkpoints/ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints for seed {seed}")
    target_path = ckpts[-1]
    if which == "best":
        import json
        log_path = f"logs/train_seed{seed}.json"
        if os.path.exists(log_path):
            data = json.load(open(log_path))
            evals = [r for r in data if isinstance(r, dict)
                     and 'train' in r and isinstance(r.get('train'), dict)]
            if evals:
                def _key(r):
                    loss_sum = 0.0
                    for dist in ("train", "compositional", "long"):
                        m = r.get(dist, {})
                        for k in ("loss_tok", "loss_depth", "loss_valid"):
                            loss_sum += float(m.get(k, 0))
                    return (r['eval_min_acc'], -loss_sum)
                best_eval = max(evals, key=_key)
                best_step = best_eval['step']
                for c in ckpts:
                    step_in_name = int(c.rsplit("_", 1)[1].split(".")[0])
                    if step_in_name == best_step:
                        target_path = c
                        break
    state = torch.load(target_path, map_location=device, weights_only=False)
    cfg_dict = state.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    m = make_model(cfg); m.load_state_dict(state["model_state"])
    return m.to(device).eval(), target_path


def model_from_aligned(template: Transformer, aligned_params: dict) -> Transformer:
    """Build a transformer from a flat param dict (deepcopy template + load)."""
    import copy
    m = copy.deepcopy(template)
    sd = m.state_dict()
    for k, v in aligned_params.items():
        if k in sd:
            sd[k] = v
    m.load_state_dict(sd, strict=False)
    return m


def kl_between_models(m1: Transformer, m2: Transformer, n_batches: int = 8,
                       batch_size: int = 256, device: str = "cpu") -> dict:
    """KL(m1 ‖ m2) averaged over a fair sample. Both models on the same device."""
    m1.to(device).eval(); m2.to(device).eval()
    rng = np.random.default_rng(99)
    task_cfg = TaskConfig(n_ctx=m1.cfg.n_ctx)
    total_kl = 0.0; total_n = 0
    per_head = {"tok": 0.0, "depth": 0.0, "valid": 0.0}
    per_head_n = {"tok": 0, "depth": 0, "valid": 0}
    with torch.no_grad():
        for _ in range(n_batches):
            b = sample_batch(batch_size, task_cfg, rng, length_range=(2, 48))
            tok = b["tok"].to(device); mask = b["mask"].to(device).double()
            o1 = m1(tok); o2 = m2(tok)
            for k in ("tok", "depth", "valid"):
                logp = F.log_softmax(o1[k].double(), dim=-1)
                logq = F.log_softmax(o2[k].double(), dim=-1)
                p = logp.exp()
                kl = (p * (logp - logq)).sum(dim=-1)
                kl = (kl * mask).sum().item()
                n = mask.sum().item()
                per_head[k] += kl; per_head_n[k] += n
                total_kl += kl; total_n += n
    return {
        "mean_kl": total_kl / max(total_n, 1),
        "per_head": {k: per_head[k] / max(per_head_n[k], 1) for k in per_head},
    }


def cross_seed_bar_C(m_aligned: Transformer, m_primary: Transformer,
                      device: str = "cuda", ig_steps: int = 32) -> dict:
    """Cross-seed Bar C: predicted (IG on primary) vs measured (mean-ablation on
    aligned_k) should be highly correlated when the seeds compute the same
    function via the same mechanism."""
    from ..analysis.lens3_causal import integrated_gradients_patch
    means_aligned = mean_activations(m_aligned, n_seqs=1024, device=device)
    means_primary = mean_activations(m_primary, n_seqs=1024, device=device)

    rng = np.random.default_rng(7)
    task_cfg = TaskConfig(n_ctx=m_aligned.cfg.n_ctx)
    batch = sample_batch(256, task_cfg, rng, length_range=(2, 48))
    batch = {k: v.to(device) for k, v in batch.items()}

    ablations_aligned = component_ablation_table(m_aligned, batch, means_aligned, device=device)["components"]
    pred_primary = integrated_gradients_patch(m_primary, batch, means_primary,
                                                n_steps=ig_steps, per_component=True,
                                                device=device)
    ablations_primary = component_ablation_table(m_primary, batch, means_primary, device=device)["components"]
    keys = sorted(set(ablations_aligned) & set(pred_primary) & set(ablations_primary))
    measured = [ablations_aligned[k] for k in keys]
    predicted = [pred_primary[k] for k in keys]
    measured_primary = [ablations_primary[k] for k in keys]
    if len(keys) < 2 or np.std(measured) < 1e-12 or np.std(predicted) < 1e-12:
        r = float("nan")
    else:
        r = float(np.corrcoef(predicted, measured)[0, 1])
    r_self = float(np.corrcoef(measured_primary, measured)[0, 1]) \
              if (len(keys) >= 2 and np.std(measured) > 1e-12 and np.std(measured_primary) > 1e-12) \
              else float("nan")
    return {
        "passed": (not np.isnan(r)) and r > 0.99,
        "tolerance": 0.99,
        "pearson_r": r,
        "pearson_r_primary_measured_vs_aligned_measured": r_self,
        "n_edges": len(keys),
    }


def cross_seed_bar_Pr(*args, **kwargs):
    """Cross-seed Bar Pr collapses to Bar C in this implementation.

    Both cross-seed bars reduce to the same operational measurement:
    predict component effects from the primary seed (via IG, equivalently
    via the primary's mean-ablation effects by the IG completeness axiom),
    measure component effects on the aligned replication seed, and report
    Pearson r over the same enumerated component set.

    The within-seed conceptual distinction (Bar C tests edge-level patch
    effects, Bar Pr tests component-level ablation losses) collapses cross-
    seed because we use component-level mean ablation as the single available
    granularity at both prediction and measurement ends. The two cross-seed
    columns are therefore bit-identical by construction and the PAPER reports
    them together for transparency (see PAPER_FINAL.md §7 disclosure and §10
    limitation).

    A meaningfully distinct cross-seed Bar Pr would compare e.g. cumulative
    top-k ablation losses (Bar Pr) vs single-component patch effects (Bar C);
    we expect this would give qualitatively the same cross-seed failure but
    have not implemented or measured it. This alias is intentional and
    documented; do not silently rename or remove without updating the paper.
    """
    return cross_seed_bar_C(*args, **kwargs)


def run_universality(primary_seed: int, replication_seed: int,
                      device: str = "cpu", out_dir: str = None,
                      n_outer: int = 6, n_starts: int = 16,
                      ig_steps: int = 32) -> dict:
    """Run all 4 bars between the primary and one replication seed."""
    print(f"[Universality] primary=seed{primary_seed} vs seed{replication_seed}",
          flush=True)
    m0, ckpt0 = load_seed(primary_seed)
    mk, ckptk = load_seed(replication_seed)
    print(f"   primary: {ckpt0}")
    print(f"   replic.: {ckptk}")

    m0f = fold_rmsnorm(m0); mkf = fold_rmsnorm(mk)
    aligned_params, info = align(mkf, m0f, n_outer=n_outer, n_starts=n_starts)
    m_aligned = model_from_aligned(mkf, aligned_params)

    # Bar P
    p0 = _params_as_dict(m0f)
    bar_P_mse = _param_mse(aligned_params, p0)
    per_tensor = {}
    for k in p0:
        d = (aligned_params[k] - p0[k]).flatten()
        per_tensor[k] = {"mse": float((d * d).mean().item()),
                          "max_abs": float(d.abs().max().item())}
    max_mse = max(v["mse"] for v in per_tensor.values())
    P_res = {
        "passed": max_mse < 1e-3,
        "tolerance": 1e-3,
        "max_per_tensor_mse": max_mse,
        "global_mse": bar_P_mse,
        "per_tensor": per_tensor,
        "mse_history": info["mse_history"],
    }
    print(f"   Bar P: max MSE={max_mse:.3e}  passed={P_res['passed']}", flush=True)

    # Bar B
    B_res = kl_between_models(m0, m_aligned, device=device)
    B_res["passed"] = B_res["mean_kl"] < 1e-4
    B_res["tolerance"] = 1e-4
    print(f"   Bar B: KL={B_res['mean_kl']:.3e}  passed={B_res['passed']}", flush=True)

    # Bar C (requires cuda)
    if device == "cuda" or torch.cuda.is_available():
        m0.cuda(); m_aligned.cuda()
        C_res = cross_seed_bar_C(m_aligned, m0, device="cuda", ig_steps=ig_steps)
        Pr_res = cross_seed_bar_Pr(m_aligned, m0, device="cuda", ig_steps=ig_steps)
        print(f"   Bar C: r={C_res['pearson_r']:.4f}  passed={C_res['passed']}", flush=True)
        print(f"   Bar Pr: r={Pr_res['pearson_r']:.4f}  passed={Pr_res['passed']}", flush=True)
    else:
        C_res = {"passed": None, "note": "skipped (cuda not available)"}
        Pr_res = {"passed": None, "note": "skipped (cuda not available)"}

    summary = {
        "primary_seed": primary_seed,
        "replication_seed": replication_seed,
        "ckpt_primary": ckpt0,
        "ckpt_replication": ckptk,
        "bars": {"B": B_res, "P": P_res, "C": C_res, "Pr": Pr_res},
        "alignment_info": {
            "head_perms": info["head_perms"],
            "mlp_perms_summary": [p[:8] for p in info["mlp_perms"]],
            "mse_history": info["mse_history"],
        },
    }
    all_passed = all(summary["bars"][b].get("passed") for b in ("B", "P", "C", "Pr")
                      if summary["bars"][b].get("passed") is not None)
    summary["all_bars_passed"] = all_passed
    summary["status"] = "aligned" if all_passed else "universality_break"

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"universality_{replication_seed}_vs_{primary_seed}.json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=int, default=0)
    parser.add_argument("--replications", default="1,2,3,4")
    parser.add_argument("--out", default="experiments/universality")
    parser.add_argument("--n_outer", type=int, default=6)
    parser.add_argument("--n_starts", type=int, default=16)
    parser.add_argument("--ig_steps", type=int, default=32)
    args = parser.parse_args()
    reps = [int(s) for s in args.replications.split(",")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summaries = []
    for r in reps:
        s = run_universality(args.primary, r, device=device, out_dir=args.out,
                              n_outer=args.n_outer, n_starts=args.n_starts,
                              ig_steps=args.ig_steps)
        summaries.append(s)
    with open(os.path.join(args.out, "universality_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2, default=str)
    print("=== Universality summary ===")
    for s in summaries:
        st = s["status"]
        bars = s["bars"]
        b_kl = bars["B"]["mean_kl"]
        p_mse = bars["P"]["max_per_tensor_mse"]
        c_r = bars["C"].get("pearson_r", "skipped")
        pr_r = bars["Pr"].get("pearson_r", "skipped")
        print(f"  seed{s['replication_seed']}: {st} | B={b_kl:.3e} P={p_mse:.3e} "
              f"C={c_r if isinstance(c_r, str) else f'{c_r:.4f}'} "
              f"Pr={pr_r if isinstance(pr_r, str) else f'{pr_r:.4f}'}")


if __name__ == "__main__":
    main()
