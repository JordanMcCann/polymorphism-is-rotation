"""Orchestrates the full post-training pipeline:

  1. For each seed: run lenses 1, 3, 4, 5 (5 = compile + cache spec for seed 0).
  2. For each seed: run all four bars vs the constructive spec from seed 0.
  3. Run universality (seed 0 vs seeds 1..4).
  4. Run lens 2 (SAEs + transcoders) on all seeds, with cross-seed stability.
  5. Run adversarial decoy verification on seed 0.
  6. Write final report.

The pipeline is restartable: each step skips work it can detect was already
done by the presence of an output file. To force a re-run, delete the
corresponding output(s).
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
    from polymorphism.analysis.lens2_saes import (
        cross_seed_feature_stability,
        run_lens2,
        run_transcoders,
    )
    from polymorphism.analysis.lens3_causal import run_lens3
    from polymorphism.analysis.lens4_polyhedral import run_lens4
    from polymorphism.analysis.lens5_rasp import compile_spec_to_model
    from polymorphism.run_lenses import load_seed_model
    from polymorphism.verification.bar_behavioral import run_bar_behavioral
    from polymorphism.verification.bar_causal import run_bar_causal
    from polymorphism.verification.bar_parametric import run_bar_parametric
    from polymorphism.verification.bar_predictive import run_bar_predictive
    from polymorphism.verification.decoy import (
        make_decoy_quantization,
        verify_bars_distinguish_decoy,
    )
    from polymorphism.verification.universality import run_universality
else:
    from .analysis.lens1_weights import run_lens1
    from .analysis.lens2_saes import cross_seed_feature_stability, run_lens2, run_transcoders
    from .analysis.lens3_causal import run_lens3
    from .analysis.lens4_polyhedral import run_lens4
    from .analysis.lens5_rasp import compile_spec_to_model
    from .run_lenses import load_seed_model
    from .verification.bar_behavioral import run_bar_behavioral
    from .verification.bar_causal import run_bar_causal
    from .verification.bar_parametric import run_bar_parametric
    from .verification.bar_predictive import run_bar_predictive
    from .verification.decoy import make_decoy_quantization, verify_bars_distinguish_decoy
    from .verification.universality import run_universality


def _skip_or_run(out_path: str, label: str, fn, force: bool = False):
    if os.path.exists(out_path) and not force:
        print(f"[skip] {label} (output exists: {out_path})", flush=True)
        with open(out_path) as f:
            return json.load(f)
    print(f"[run]  {label}", flush=True)
    t0 = time.time()
    res = fn()
    print(f"       done in {time.time()-t0:.1f}s", flush=True)
    return res


def run_lenses_for_seed(seed: int, device: str, args):
    print(f"\n=== Lenses for seed {seed} ===", flush=True)
    model, ckpt = load_seed_model(seed, device=device, which='best')
    lens_dir = f"experiments/seeds/{seed}/lens_outputs"
    os.makedirs(lens_dir, exist_ok=True)
    print(f"checkpoint: {ckpt}", flush=True)

    if not os.path.exists(os.path.join(lens_dir, "lens1.json")) or args.force:
        run_lens1(model, lens_dir)

    # Lens 5: build / cache constructive spec.
    if seed == args.primary_seed:
        spec_cache = os.path.join(lens_dir, "spec_constructive.pt")
        if not os.path.exists(spec_cache) or args.force:
            print(f"[seed {seed}] Building constructive spec...", flush=True)
            spec_model, info = compile_spec_to_model(model.cfg,
                                                       mode='constructive',
                                                       primary_seed=args.primary_seed)
            print(f"       spec built (K_mlp={info['meta'].get('mlp_K_per_layer')}, "
                  f"min_acc={info['meta'].get('min_acc_at_K'):.6f})", flush=True)

    if not os.path.exists(os.path.join(lens_dir, "lens4.json")) or args.force:
        run_lens4(model, lens_dir, n_seqs=args.lens4_seqs, device=device)

    if not os.path.exists(os.path.join(lens_dir, "lens3.json")) or args.force:
        run_lens3(model, lens_dir, n_batches=4, batch_size=256, device=device)


def run_bars_for_seed(seed: int, device: str, args):
    print(f"\n=== Bars for seed {seed} ===", flush=True)
    model, ckpt = load_seed_model(seed, device=device, which='best')
    bar_dir = f"experiments/seeds/{seed}/bar_outputs"
    os.makedirs(bar_dir, exist_ok=True)

    # Bar B
    if not os.path.exists(os.path.join(bar_dir, "bar_B.json")) or args.force:
        ckpt_b = os.path.join(bar_dir, "bar_B_checkpoint.pt")
        rB = run_bar_behavioral(model, n_samples=args.bar_b_samples,
                                  batch_size=args.bar_b_batch, device=device,
                                  progress_every=200, checkpoint_path=ckpt_b)
        with open(os.path.join(bar_dir, "bar_B.json"), "w") as f:
            json.dump(rB, f, indent=2, default=str)
        print(f"  Bar B: KL={rB['mean_kl']:.3e} passed={rB['passed']}", flush=True)

    # Bar P
    if not os.path.exists(os.path.join(bar_dir, "bar_P.json")) or args.force:
        rP = run_bar_parametric(model.cpu(), n_outer=args.n_outer,
                                   n_starts=args.n_starts,
                                   spec_mode='constructive',
                                   primary_seed=args.primary_seed)
        model.to(device)
        with open(os.path.join(bar_dir, "bar_P.json"), "w") as f:
            json.dump(rP, f, indent=2, default=str)
        print(f"  Bar P: max MSE={rP['max_per_tensor_mse']:.3e} passed={rP['passed']}", flush=True)

    # Bar C
    if not os.path.exists(os.path.join(bar_dir, "bar_C.json")) or args.force:
        rC = run_bar_causal(model, n_seqs=2048, device=device, batch_size=256,
                              ig_steps=args.ig_steps, spec_mode='constructive',
                              primary_seed=args.primary_seed)
        with open(os.path.join(bar_dir, "bar_C.json"), "w") as f:
            json.dump(rC, f, indent=2, default=str)
        print(f"  Bar C: r={rC['pearson_r']:.4f} passed={rC['passed']} "
              f"(r_ig={rC.get('pearson_r_ig')})", flush=True)

    # Bar Pr
    if not os.path.exists(os.path.join(bar_dir, "bar_Pr.json")) or args.force:
        rPr = run_bar_predictive(model, n_seqs=2048, device=device,
                                    ig_steps=args.ig_steps, spec_mode='constructive',
                                    primary_seed=args.primary_seed)
        with open(os.path.join(bar_dir, "bar_Pr.json"), "w") as f:
            json.dump(rPr, f, indent=2, default=str)
        print(f"  Bar Pr: r={rPr['pearson_r']:.4f} passed={rPr['passed']}", flush=True)


def run_universality_all(args):
    print(f"\n=== Universality (primary={args.primary_seed}) ===", flush=True)
    out_dir = "experiments/universality"
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reps = [s for s in args.seeds if s != args.primary_seed]
    for r in reps:
        out_path = os.path.join(out_dir, f"universality_{r}_vs_{args.primary_seed}.json")
        if os.path.exists(out_path) and not args.force:
            print(f"[skip] universality seed{r} vs seed{args.primary_seed}", flush=True)
            continue
        run_universality(args.primary_seed, r, device=device, out_dir=out_dir,
                          n_outer=args.n_outer, n_starts=args.n_starts,
                          ig_steps=args.ig_steps)


def run_lens2_all(args, device: str):
    print("\n=== Lens 2 (SAEs + transcoders) ===", flush=True)
    for seed in args.seeds:
        lens_dir = f"experiments/seeds/{seed}/lens_outputs"
        sae_marker = os.path.join(lens_dir, "lens2_summary.json")
        if os.path.exists(sae_marker) and not args.force:
            print(f"[skip] Lens 2 seed {seed}", flush=True)
            continue
        model, _ = load_seed_model(seed, device=device, which='best')
        run_lens2(model, lens_dir, expansions=tuple(args.expansions),
                  n_seqs=args.sae_seqs, n_steps=args.sae_steps, device=device)
        run_transcoders(model, lens_dir, expansions=tuple(args.expansions),
                        n_seqs=args.sae_seqs, n_steps=args.sae_steps, device=device)
    # Cross-seed stability: compare SAE features between seeds
    if len(args.seeds) >= 2:
        print("\n=== Cross-seed SAE feature stability ===", flush=True)
        primary = args.primary_seed
        stability = {}
        for s in args.seeds:
            if s == primary:
                continue
            pair_stat = {}
            for site in ("resid_post_0", "resid_post_1", "resid_pre_2"):
                expand = args.expansions[0]
                fa = f"experiments/seeds/{primary}/lens_outputs/sae_{site}_x{expand}.pt"
                fb = f"experiments/seeds/{s}/lens_outputs/sae_{site}_x{expand}.pt"
                if os.path.exists(fa) and os.path.exists(fb):
                    a = torch.load(fa, map_location='cpu', weights_only=False)
                    b = torch.load(fb, map_location='cpu', weights_only=False)
                    stab = cross_seed_feature_stability(a["state"], b["state"], threshold=0.5)
                    pair_stat[site] = stab
            stability[f"seed{s}_vs_seed{primary}"] = pair_stat
        with open("logs/sae_cross_seed_stability.json", "w") as f:
            json.dump(stability, f, indent=2, default=str)
        print(json.dumps({k: {s: round(v["fraction_stable"], 3)
                               for s, v in pairs.items()}
                            for k, pairs in stability.items()}, indent=2))


def run_decoy(args, device: str):
    print("\n=== Adversarial decoy verification ===", flush=True)
    out_path = "logs/decoy_verification.json"
    if os.path.exists(out_path) and not args.force:
        print(f"[skip] {out_path} exists", flush=True)
        return
    real, _ = load_seed_model(args.primary_seed, device=device, which='best')
    decoy = make_decoy_quantization(real)
    res = verify_bars_distinguish_decoy(real, decoy,
                                          primary_seed=args.primary_seed,
                                          device=device,
                                          bars=("B", "P"))
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"  decoy_rejected: {res['decoy_rejected']} (by: {res['decoy_rejected_by']})", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--primary_seed", type=int, default=0)
    parser.add_argument("--bar_b_samples", type=int, default=100_000_000)
    parser.add_argument("--bar_b_batch", type=int, default=2048)
    parser.add_argument("--n_outer", type=int, default=6)
    parser.add_argument("--n_starts", type=int, default=16)
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--lens4_seqs", type=int, default=8192)
    parser.add_argument("--sae_seqs", type=int, default=4096)
    parser.add_argument("--sae_steps", type=int, default=4000)
    parser.add_argument("--expansions", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--skip", default="",
                        help="Comma-separated stages to skip: lenses,bars,universality,sae,decoy,report")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even when outputs exist")
    args = parser.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(",")]
    skip = set(args.skip.split(",")) if args.skip else set()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, seeds={args.seeds}, primary={args.primary_seed}", flush=True)

    if "lenses" not in skip:
        # Lenses for primary seed first (so spec is cached before bars on other seeds)
        ordered = [args.primary_seed] + [s for s in args.seeds if s != args.primary_seed]
        for s in ordered:
            run_lenses_for_seed(s, device, args)

    if "bars" not in skip:
        # Bars: primary seed first (spec already cached), then others
        ordered = [args.primary_seed] + [s for s in args.seeds if s != args.primary_seed]
        for s in ordered:
            run_bars_for_seed(s, device, args)

    if "universality" not in skip:
        run_universality_all(args)

    if "sae" not in skip:
        run_lens2_all(args, device)

    if "decoy" not in skip:
        run_decoy(args, device)

    if "report" not in skip:
        print("\n=== Final report ===", flush=True)
        # Re-invoke as subprocess so its argparse doesn't conflict
        import subprocess
        subprocess.run([sys.executable, "-m", "polymorphism.write_logs",
                         "--seeds", ",".join(map(str, args.seeds))],
                        check=False)
        subprocess.run([sys.executable, "-m", "polymorphism.final_report",
                         "--seeds", ",".join(map(str, args.seeds)),
                         "--primary", str(args.primary_seed)],
                        check=False)


if __name__ == "__main__":
    main()
