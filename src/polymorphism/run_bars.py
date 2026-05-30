"""Orchestrate the four-bar verification on one seed.

Usage:
    python -m polymorphism.run_bars --seed 0 [--n_samples_b 100000000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from polymorphism.run_lenses import load_seed_model
    from polymorphism.verification.bar_behavioral import run_bar_behavioral
    from polymorphism.verification.bar_causal import run_bar_causal
    from polymorphism.verification.bar_parametric import run_bar_parametric
    from polymorphism.verification.bar_predictive import run_bar_predictive
else:
    from .run_lenses import load_seed_model
    from .verification.bar_behavioral import run_bar_behavioral
    from .verification.bar_causal import run_bar_causal
    from .verification.bar_parametric import run_bar_parametric
    from .verification.bar_predictive import run_bar_predictive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0,
                        help="The seed to run the bars on.")
    parser.add_argument("--primary_seed", type=int, default=0,
                        help="The seed used to derive the constructive spec.")
    parser.add_argument("--n_samples_b", type=int, default=100_000_000)
    parser.add_argument("--batch_size_b", type=int, default=2048)
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--n_starts", type=int, default=16,
                        help="Multi-start count for parametric alignment.")
    parser.add_argument("--n_outer", type=int, default=6)
    parser.add_argument("--spec_mode", default="constructive",
                        choices=["constructive", "handrolled"])
    parser.add_argument("--bars", default="B,P,C,Pr")
    args = parser.parse_args()

    selected = set(args.bars.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_path = load_seed_model(args.seed, device=device, which='best')
    out_dir = f"experiments/seeds/{args.seed}/bar_outputs"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Running bars on {ckpt_path}  (spec={args.spec_mode}, "
          f"primary_seed={args.primary_seed})", flush=True)
    results = {}

    if "B" in selected:
        print("=== Bar B (behavioral) ===", flush=True)
        t0 = time.time()
        ckpt_path_b = os.path.join(out_dir, "bar_B_checkpoint.pt")
        r = run_bar_behavioral(model, n_samples=args.n_samples_b,
                                 batch_size=args.batch_size_b,
                                 device=device, progress_every=200,
                                 checkpoint_path=ckpt_path_b)
        results["B"] = r
        with open(os.path.join(out_dir, "bar_B.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"Bar B: KL={r['mean_kl']:.3e}  passed={r['passed']}  ({time.time()-t0:.1f}s)",
              flush=True)
    if "P" in selected:
        print("=== Bar P (parametric) ===", flush=True)
        t0 = time.time()
        r = run_bar_parametric(model.cpu(), n_outer=args.n_outer,
                                 n_starts=args.n_starts,
                                 spec_mode=args.spec_mode,
                                 primary_seed=args.primary_seed)
        model.to(device)
        results["P"] = r
        with open(os.path.join(out_dir, "bar_P.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"Bar P: max MSE={r['max_per_tensor_mse']:.3e}  passed={r['passed']}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    if "C" in selected:
        print("=== Bar C (causal) ===", flush=True)
        t0 = time.time()
        r = run_bar_causal(model, n_seqs=4096, device=device,
                             ig_steps=args.ig_steps, spec_mode=args.spec_mode,
                             primary_seed=args.primary_seed)
        results["C"] = r
        with open(os.path.join(out_dir, "bar_C.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"Bar C: Pearson r={r['pearson_r']:.4f}  passed={r['passed']}  "
              f"r_ig={r.get('pearson_r_ig')}  r_attr={r.get('pearson_r_attribution')}  "
              f"({time.time()-t0:.1f}s)", flush=True)
    if "Pr" in selected:
        print("=== Bar Pr (predictive) ===", flush=True)
        t0 = time.time()
        r = run_bar_predictive(model, n_seqs=2048, device=device,
                                  ig_steps=args.ig_steps, spec_mode=args.spec_mode,
                                  primary_seed=args.primary_seed)
        results["Pr"] = r
        with open(os.path.join(out_dir, "bar_Pr.json"), "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"Bar Pr: Pearson r={r['pearson_r']:.4f}  passed={r['passed']}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk in
                       ("passed", "mean_kl", "max_per_tensor_mse", "pearson_r",
                        "tolerance", "n_edges", "n_components")}
                    for k, v in results.items()},
                  f, indent=2, default=str)
    print("All bars complete.", flush=True)


if __name__ == "__main__":
    main()
