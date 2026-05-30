"""End-to-end analysis: run all lenses, all bars, and build the artifact
for the primary seed (and replication seeds if requested).

Usage:
    python -m polymorphism.run_all --seed 0                # full analysis on seed 0
    python -m polymorphism.run_all --seed 0,1,2,3,4         # also replications
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
    from polymorphism.analysis.lens1_weights import run_lens1
    from polymorphism.analysis.lens2_saes import run_lens2, run_transcoders
    from polymorphism.analysis.lens3_causal import run_lens3
    from polymorphism.analysis.lens4_polyhedral import run_lens4
    from polymorphism.analysis.lens5_rasp import run_lens5
    from polymorphism.artifact_builder import build as build_artifact
    from polymorphism.run_lenses import load_seed_model
    from polymorphism.verification.bar_behavioral import run_bar_behavioral
    from polymorphism.verification.bar_causal import run_bar_causal
    from polymorphism.verification.bar_parametric import run_bar_parametric
    from polymorphism.verification.bar_predictive import run_bar_predictive
else:
    from .analysis.lens1_weights import run_lens1
    from .analysis.lens2_saes import run_lens2, run_transcoders
    from .analysis.lens3_causal import run_lens3
    from .analysis.lens4_polyhedral import run_lens4
    from .analysis.lens5_rasp import run_lens5
    from .artifact_builder import build as build_artifact
    from .run_lenses import load_seed_model
    from .verification.bar_behavioral import run_bar_behavioral
    from .verification.bar_causal import run_bar_causal
    from .verification.bar_parametric import run_bar_parametric
    from .verification.bar_predictive import run_bar_predictive


def run_seed(seed: int, args, device: str) -> dict:
    model, ckpt = load_seed_model(seed, device=device)
    lens_dir = f"experiments/seeds/{seed}/lens_outputs"
    bar_dir = f"experiments/seeds/{seed}/bar_outputs"
    os.makedirs(lens_dir, exist_ok=True)
    os.makedirs(bar_dir, exist_ok=True)
    print(f"=== Seed {seed} -- {ckpt} ===", flush=True)
    summary = {"seed": seed, "ckpt": ckpt, "lenses": {}, "bars": {}}

    # Lens 1 -- weight decomposition
    t0 = time.time()
    print(f"[Seed {seed}] Lens 1 (weight decomp)...", flush=True)
    run_lens1(model, lens_dir)
    summary["lenses"]["1"] = {"elapsed_s": time.time() - t0}

    # Lens 5 -- compiled spec (cheap)
    t0 = time.time()
    print(f"[Seed {seed}] Lens 5 (compiled spec)...", flush=True)
    run_lens5(model, lens_dir)
    summary["lenses"]["5"] = {"elapsed_s": time.time() - t0}

    # Lens 4 -- polyhedral
    t0 = time.time()
    print(f"[Seed {seed}] Lens 4 (polyhedral)...", flush=True)
    run_lens4(model, lens_dir, n_seqs=args.lens4_seqs, device=device)
    summary["lenses"]["4"] = {"elapsed_s": time.time() - t0}

    # Lens 3 -- causal
    t0 = time.time()
    print(f"[Seed {seed}] Lens 3 (causal interventions)...", flush=True)
    run_lens3(model, lens_dir, n_batches=4, batch_size=256, device=device)
    summary["lenses"]["3"] = {"elapsed_s": time.time() - t0}

    # Lens 2 -- SAEs
    if args.run_lens2:
        t0 = time.time()
        print(f"[Seed {seed}] Lens 2 (SAEs)...", flush=True)
        run_lens2(model, lens_dir, expansions=tuple(args.expansions),
                  n_seqs=args.sae_seqs, n_steps=args.sae_steps, device=device)
        run_transcoders(model, lens_dir, expansions=tuple(args.expansions),
                        n_seqs=args.sae_seqs, n_steps=args.sae_steps, device=device)
        summary["lenses"]["2"] = {"elapsed_s": time.time() - t0}

    # Bars
    t0 = time.time()
    print(f"[Seed {seed}] Bar B (behavioral)...", flush=True)
    bar_b_ckpt = os.path.join(bar_dir, "bar_B_checkpoint.pt")
    rB = run_bar_behavioral(model, n_samples=args.bar_b_samples,
                            batch_size=args.bar_b_batch, device=device,
                            progress_every=200,
                            checkpoint_path=bar_b_ckpt)
    summary["bars"]["B"] = {k: rB[k] for k in ("mean_kl", "passed", "tolerance", "per_head_kl", "per_distribution")}
    with open(os.path.join(bar_dir, "bar_B.json"), "w") as f:
        json.dump(rB, f, indent=2, default=str)
    print(f"   Bar B: KL={rB['mean_kl']:.3e}  passed={rB['passed']}  ({time.time()-t0:.1f}s)",
          flush=True)

    t0 = time.time()
    print(f"[Seed {seed}] Bar P (parametric)...", flush=True)
    rP = run_bar_parametric(model.cpu(), n_outer=args.bar_p_outer,
                              n_starts=args.bar_p_starts,
                              spec_mode=args.spec_mode,
                              primary_seed=args.primary_seed)
    model.to(device)
    summary["bars"]["P"] = {"passed": rP["passed"], "max_per_tensor_mse": rP["max_per_tensor_mse"],
                            "global_mse": rP["global_mse"], "tolerance": rP["tolerance"]}
    with open(os.path.join(bar_dir, "bar_P.json"), "w") as f:
        json.dump(rP, f, indent=2, default=str)
    print(f"   Bar P: max MSE={rP['max_per_tensor_mse']:.3e}  passed={rP['passed']}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    print(f"[Seed {seed}] Bar C (causal)...", flush=True)
    rC = run_bar_causal(model, n_seqs=args.bar_seqs, device=device, batch_size=256,
                         ig_steps=args.ig_steps, spec_mode=args.spec_mode,
                         primary_seed=args.primary_seed)
    summary["bars"]["C"] = {k: rC[k] for k in ("pearson_r", "pearson_r_ig",
                                                "pearson_r_attribution",
                                                "pearson_r_spec_ablation",
                                                "passed", "tolerance")}
    with open(os.path.join(bar_dir, "bar_C.json"), "w") as f:
        json.dump(rC, f, indent=2, default=str)
    print(f"   Bar C: best r={rC['pearson_r']:.4f}  passed={rC['passed']}  "
          f"r_ig={rC.get('pearson_r_ig')}  ({time.time()-t0:.1f}s)",
          flush=True)

    t0 = time.time()
    print(f"[Seed {seed}] Bar Pr (predictive)...", flush=True)
    rPr = run_bar_predictive(model, n_seqs=args.bar_seqs, device=device,
                                ig_steps=args.ig_steps, spec_mode=args.spec_mode,
                                primary_seed=args.primary_seed)
    summary["bars"]["Pr"] = {k: rPr[k] for k in ("pearson_r", "pearson_r_ig",
                                                  "passed", "tolerance")}
    with open(os.path.join(bar_dir, "bar_Pr.json"), "w") as f:
        json.dump(rPr, f, indent=2, default=str)
    print(f"   Bar Pr: r={rPr['pearson_r']:.4f}  passed={rPr['passed']}  ({time.time()-t0:.1f}s)",
          flush=True)

    with open(os.path.join(bar_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--primary_seed", type=int, default=0,
                        help="Seed from which the constructive spec is built.")
    parser.add_argument("--spec_mode", default="constructive",
                        choices=["constructive", "handrolled"])
    parser.add_argument("--bar_b_samples", type=int, default=100_000_000,
                        help="Sample count for Bar B (directive specifies 1e8).")
    parser.add_argument("--bar_b_batch", type=int, default=2048)
    parser.add_argument("--bar_p_outer", type=int, default=6)
    parser.add_argument("--bar_p_starts", type=int, default=16)
    parser.add_argument("--bar_seqs", type=int, default=2048)
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--lens4_seqs", type=int, default=8192)
    parser.add_argument("--sae_seqs", type=int, default=4096)
    parser.add_argument("--sae_steps", type=int, default=4000)
    parser.add_argument("--expansions", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--run_lens2", action="store_true", default=True)
    parser.add_argument("--build_artifact", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    all_summaries = []

    for seed in seeds:
        summary = run_seed(seed, args, device)
        all_summaries.append(summary)
        with open(f"logs/run_all_seed{seed}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

    if args.build_artifact:
        print("=== Building artifact ===", flush=True)
        build_artifact(seeds[0])
        print("Artifact built.", flush=True)


if __name__ == "__main__":
    main()
