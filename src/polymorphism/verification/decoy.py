"""Adversarial decoy: a model that behaves like a real seed but has
algorithmically distinct internals.

Used to verify that the four bars discriminate between the genuine
mechanism and a "shape-of-the-network looks right but the algorithm
is different" decoy. This addresses Attack 1 in
`logs/adversarial_review.md`: the worry that all four bars might pass
simultaneously for a model that merely *looks* close to the spec but
does the wrong computation.

The verification predicate (`verify_bars_distinguish_decoy`) requires
that the decoy fails at least one bar that the real model passes.
"""

from __future__ import annotations

import json
import os
from copy import deepcopy

import torch

from ..model import Transformer
from ..task import N_DEPTH


def make_decoy_quantization(model_seed: Transformer) -> Transformer:
    """Decoy: collapse adjacent depth bins in W_U_depth so the model returns
    a 5-bin quantisation of the depth labels instead of the original 9.

    The bracket-token head and the validity head are unchanged, so the decoy
    will pass Bar B *only on the token head*. The depth head is intentionally
    mis-aligned with the trained model's distribution, ensuring Bar B fails.

    The decoy is otherwise weight-similar to the trained model — its W_U_depth
    differs by only a few row averages — so it could fool a naive parametric
    comparison that doesn't reckon with the depth-output dimension.
    """
    decoy = deepcopy(model_seed)
    W = decoy.W_U_depth.data.clone()
    for d in range(0, N_DEPTH - 1, 2):
        avg = (W[d] + W[d + 1]) / 2
        W[d] = avg
        W[d + 1] = avg
    decoy.W_U_depth.data = W
    return decoy


def make_decoy_swap_layers(model_seed: Transformer) -> Transformer:
    """Decoy 2: swap layers 0 and 1.

    The architecture is symmetric in layer index, so this is a valid model;
    it computes a totally different algorithm (layer-1 features compute first,
    layer-0 features compute second). The decoy may still produce some valid
    outputs but its mechanism is wrong; Bar P should reject it since the
    per-tensor MSE is large under any allowed permutation.
    """
    decoy = deepcopy(model_seed)
    b0, b1 = decoy.blocks[0], decoy.blocks[1]
    # Swap by deepcopy to avoid aliasing
    sd0 = b0.state_dict()
    sd1 = b1.state_dict()
    b0.load_state_dict(sd1)
    b1.load_state_dict(sd0)
    return decoy


def verify_bars_distinguish_decoy(model_real: Transformer,
                                   model_decoy: Transformer,
                                   primary_seed: int = 0,
                                   device: str = "cuda",
                                   bars: tuple[str, ...] = ("B", "P", "C", "Pr"),
                                   bar_B_samples: int = 100_000,
                                   ) -> dict:
    """Compute the requested bars for (model_real vs spec) and (model_decoy vs spec).

    Returns the per-bar comparison plus an overall "decoy_rejected" flag (True
    iff the decoy fails at least one bar that the real model passes).

    bar_B_samples is reduced to 1e5 (not 1e8) here since this is a sanity
    check; the full-scale Bar B is run separately by run_bars.
    """
    from .bar_behavioral import run_bar_behavioral
    from .bar_causal import run_bar_causal
    from .bar_parametric import run_bar_parametric
    from .bar_predictive import run_bar_predictive

    results = {"real": {}, "decoy": {}, "decoy_rejected_by": []}
    real_pass = {}
    decoy_pass = {}
    if "B" in bars:
        rb = run_bar_behavioral(model_real, n_samples=bar_B_samples,
                                 batch_size=1024, device=device)
        db = run_bar_behavioral(model_decoy, n_samples=bar_B_samples,
                                 batch_size=1024, device=device)
        results["real"]["B"] = rb; results["decoy"]["B"] = db
        real_pass["B"] = rb["passed"]; decoy_pass["B"] = db["passed"]
    if "P" in bars:
        rp = run_bar_parametric(model_real, primary_seed=primary_seed)
        dp = run_bar_parametric(model_decoy, primary_seed=primary_seed)
        results["real"]["P"] = {"passed": rp["passed"],
                                "max_per_tensor_mse": rp["max_per_tensor_mse"]}
        results["decoy"]["P"] = {"passed": dp["passed"],
                                 "max_per_tensor_mse": dp["max_per_tensor_mse"]}
        real_pass["P"] = rp["passed"]; decoy_pass["P"] = dp["passed"]
    if "C" in bars:
        rc = run_bar_causal(model_real, device=device, primary_seed=primary_seed)
        dc = run_bar_causal(model_decoy, device=device, primary_seed=primary_seed)
        results["real"]["C"] = {"passed": rc["passed"], "pearson_r": rc["pearson_r"]}
        results["decoy"]["C"] = {"passed": dc["passed"], "pearson_r": dc["pearson_r"]}
        real_pass["C"] = rc["passed"]; decoy_pass["C"] = dc["passed"]
    if "Pr" in bars:
        rpr = run_bar_predictive(model_real, device=device, primary_seed=primary_seed)
        dpr = run_bar_predictive(model_decoy, device=device, primary_seed=primary_seed)
        results["real"]["Pr"] = {"passed": rpr["passed"], "pearson_r": rpr["pearson_r"]}
        results["decoy"]["Pr"] = {"passed": dpr["passed"], "pearson_r": dpr["pearson_r"]}
        real_pass["Pr"] = rpr["passed"]; decoy_pass["Pr"] = dpr["passed"]
    for b in bars:
        if real_pass.get(b, False) and not decoy_pass.get(b, False):
            results["decoy_rejected_by"].append(b)
    results["decoy_rejected"] = len(results["decoy_rejected_by"]) > 0
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decoy_kind", choices=["quantization", "swap_layers"],
                        default="quantization")
    parser.add_argument("--out", default="logs/decoy_verification.json")
    args = parser.parse_args()
    from ..run_lenses import load_seed_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real, _ = load_seed_model(args.seed, device=device)
    if args.decoy_kind == "quantization":
        decoy = make_decoy_quantization(real)
    else:
        decoy = make_decoy_swap_layers(real)
    res = verify_bars_distinguish_decoy(real, decoy, primary_seed=args.seed,
                                          device=device,
                                          bars=("B", "P", "C", "Pr"))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps({"decoy_kind": args.decoy_kind,
                       "decoy_rejected": res["decoy_rejected"],
                       "decoy_rejected_by": res["decoy_rejected_by"]}, indent=2))


if __name__ == "__main__":
    main()
