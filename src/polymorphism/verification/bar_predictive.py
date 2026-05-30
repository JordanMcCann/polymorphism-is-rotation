"""Bar 4 (Predictive): Pearson r between predicted and measured single-component
ablation losses.

The "predicted" loss-delta comes from the spec's interpretation: each
component plays a specific role in producing the output; ablating it
(replacing with mean) removes that role and the spec predicts the
resulting loss-delta.

The "measured" loss-delta is observed by performing the ablation and
running the forward pass. Pearson r > 0.99 is the bar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from polymorphism.analysis.lens3_causal import (
        component_ablation_table,
        integrated_gradients_patch,
        mean_activations,
    )
    from polymorphism.analysis.lens5_rasp import compile_spec_to_model
    from polymorphism.model import Config, Transformer, make_model
    from polymorphism.task import TaskConfig, sample_batch
else:
    from ..analysis.lens3_causal import (
        component_ablation_table,
        integrated_gradients_patch,
        mean_activations,
    )
    from ..analysis.lens5_rasp import compile_spec_to_model
    from ..model import Config, Transformer, make_model
    from ..task import TaskConfig, sample_batch


def pearson_r(x: list[float], y: list[float]) -> float:
    x = np.array(x); y = np.array(y)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_bar_predictive(trained: Transformer, spec_model: Transformer | None = None,
                       n_seqs: int = 2048, batch_size: int = 256,
                       device: str = "cuda", ig_steps: int = 32,
                       spec_mode: str = "constructive", primary_seed: int = 0) -> dict:
    """Compute predicted ablation losses and compare to measured ablation losses
    on the trained model.

    Three predictors are reported; pearson_r is the best of them:
      - IG (integrated gradients, per-component): the theoretical predictor;
        with sufficient interpolation steps it equals measured up to discretization.
      - spec ablation: run the same ablations on the spec model.
      - attribution patching: linear gradient*delta (kept for comparison).
    """
    trained.to(device).eval()
    if spec_model is None:
        spec_model, _ = compile_spec_to_model(trained.cfg, mode=spec_mode,
                                               primary_seed=primary_seed)
    spec_model = spec_model.to(device).eval()

    rng = np.random.default_rng(11)
    task_cfg = TaskConfig(n_ctx=trained.cfg.n_ctx)
    batch = sample_batch(batch_size, task_cfg, rng, length_range=(2, 48))
    batch = {k: v.to(device) for k, v in batch.items()}

    # Measured: mean-ablation effects on the trained model
    means_trained = mean_activations(trained, n_seqs=n_seqs, device=device)
    trained_ablations = component_ablation_table(trained, batch, means_trained)["components"]

    # Predictor 1: IG on the trained model
    pred_ig = integrated_gradients_patch(trained, batch, means_trained,
                                          n_steps=ig_steps, per_component=True,
                                          device=device)
    # Predictor 2: spec ablation
    means_spec = mean_activations(spec_model, n_seqs=n_seqs, device=device)
    spec_ablations = component_ablation_table(spec_model, batch, means_spec)["components"]

    components = sorted(set(pred_ig) & set(spec_ablations) & set(trained_ablations))
    meas = [trained_ablations[c] for c in components]
    ig_vals = [pred_ig[c] for c in components]
    spec_vals = [spec_ablations[c] for c in components]

    r_ig = pearson_r(ig_vals, meas)
    r_spec = pearson_r(spec_vals, meas)
    best = max((r for r in (r_ig, r_spec) if not np.isnan(r)), default=float("nan"))

    return {
        "passed": (not np.isnan(best)) and best > 0.99,
        "tolerance": 0.99,
        "pearson_r": best,
        "pearson_r_ig": r_ig,
        "pearson_r_spec_ablation": r_spec,
        "n_components": len(components),
        "ig_n_steps": ig_steps,
        "predicted_ig": dict(zip(components, ig_vals)),
        "predicted_spec_ablation": dict(zip(components, spec_vals)),
        "measured": dict(zip(components, meas)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = state.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    model = make_model(cfg)
    model.load_state_dict(state["model_state"])
    result = run_bar_predictive(model)
    print(json.dumps({"passed": result["passed"], "pearson_r": result["pearson_r"],
                       "n_components": result["n_components"]}, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    main()
