"""Bar 2 (Parametric): per-entry weight MSE after symmetry alignment.

For each weight tensor, find the symmetry-group element g that minimises
||g.W_spec - W_trained||^2 and report the per-entry MSE.

The symmetry search uses polymorphism.symmetry_search.align (coordinate descent).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from polymorphism.analysis.lens5_rasp import compile_spec_to_model
    from polymorphism.model import Config, Transformer, make_model
    from polymorphism.rmsnorm_fold import fold_rmsnorm
    from polymorphism.symmetry_search import _params_as_dict, align
else:
    from ..analysis.lens5_rasp import compile_spec_to_model
    from ..model import Config, Transformer, make_model
    from ..rmsnorm_fold import fold_rmsnorm
    from ..symmetry_search import _params_as_dict, align


def per_tensor_mse(p_aligned: dict, p_ref: dict) -> dict:
    out = {}
    for k in p_ref:
        d = (p_aligned[k] - p_ref[k]).flatten()
        out[k] = {
            "mse": float((d * d).mean().item()),
            "max_abs": float(d.abs().max().item()),
            "n": int(d.numel()),
            "ref_norm": float(p_ref[k].norm().item()),
        }
    return out


def run_bar_parametric(trained_model: Transformer,
                       n_outer: int = 6, n_starts: int = 16,
                       spec_mode: str = "constructive",
                       primary_seed: int = 0) -> dict:
    """Align trained model to spec model, report per-tensor MSE."""
    spec_model, basis = compile_spec_to_model(trained_model.cfg,
                                               mode=spec_mode,
                                               primary_seed=primary_seed)
    # Ensure both models on the same device for symmetry alignment
    # (the trained model is typically moved to CPU before this call; the spec
    # may have been built/loaded on cuda — keep them on CPU for the alignment
    # which is much memory-friendlier on the small d_model=64 weights).
    trained_model = trained_model.cpu()
    spec_model = spec_model.cpu()
    spec_folded = fold_rmsnorm(spec_model)
    trained_folded = fold_rmsnorm(trained_model)
    aligned_p, info = align(trained_folded, spec_folded,
                             n_outer=n_outer, n_starts=n_starts)
    ref_p = _params_as_dict(spec_folded)
    per_tensor = per_tensor_mse(aligned_p, ref_p)
    max_mse = max(v["mse"] for v in per_tensor.values())
    avg_mse = sum(v["mse"] * v["n"] for v in per_tensor.values()) / max(
        sum(v["n"] for v in per_tensor.values()), 1
    )
    return {
        "passed": max_mse < 1e-3,
        "tolerance": 1e-3,
        "max_per_tensor_mse": max_mse,
        "global_mse": avg_mse,
        "per_tensor": per_tensor,
        "alignment_mse_history": info["mse_history"],
        "alignment_best_mse": info.get("best_mse", info["mse_history"][-1]),
        "head_perms": info["head_perms"],
        "mlp_perms_summary": [
            {"layer": L, "head": p[:8]} for L, p in enumerate(info["mlp_perms"])
        ],
        "n_starts": info.get("n_starts", 1),
        "best_start_index": info.get("best_start_index", 0),
        "spec_mode": spec_mode,
        "primary_seed": primary_seed,
        "basis": basis,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--n_outer", type=int, default=4)
    args = parser.parse_args()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = state.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in Config.__dataclass_fields__})
    model = make_model(cfg)
    model.load_state_dict(state["model_state"])
    result = run_bar_parametric(model, n_outer=args.n_outer)
    print("max per-tensor MSE:", result["max_per_tensor_mse"])
    print("passed:", result["passed"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
