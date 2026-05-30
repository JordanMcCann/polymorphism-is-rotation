"""Orchestrate the 5-lens analysis on one seed.

Usage:
    python -m polymorphism.run_lenses --seed 0
"""

from __future__ import annotations

import argparse
import glob
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
    from polymorphism.model import Config, make_model
else:
    from .analysis.lens1_weights import run_lens1
    from .analysis.lens2_saes import run_lens2, run_transcoders
    from .analysis.lens3_causal import run_lens3
    from .analysis.lens4_polyhedral import run_lens4
    from .analysis.lens5_rasp import run_lens5
    from .model import Config, make_model


def load_seed_model(seed: int, device: str = "cuda", which: str = "best"):
    """Load a checkpoint for `seed`. which='best' picks the highest min_acc
    in the eval log, breaking ties by lowest total loss (so we prefer
    high-confidence checkpoints when several reach the same accuracy);
    which='last' picks the final checkpoint."""
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
                    # Highest min_acc, then lowest total per-head loss across distributions
                    loss_sum = 0.0
                    for dist in ("train", "compositional", "long"):
                        m = r.get(dist, {})
                        for k in ("loss_tok", "loss_depth", "loss_valid"):
                            loss_sum += float(m.get(k, 0))
                    return (r['eval_min_acc'], -loss_sum)
                best_eval = max(evals, key=_key)
                best_step = best_eval['step']
                # find the nearest checkpoint
                for c in ckpts:
                    step_in_name = int(c.rsplit("_", 1)[1].split(".")[0])
                    if step_in_name == best_step:
                        target_path = c
                        break
                else:
                    # nearest below
                    target_path = max(ckpts, key=lambda c:
                                       int(c.rsplit("_", 1)[1].split(".")[0])
                                       if int(c.rsplit("_", 1)[1].split(".")[0]) <= best_step
                                       else -1)
    state = torch.load(target_path, map_location=device, weights_only=False)
    cfg_dict = state.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    model = make_model(cfg)
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    return model, target_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lenses", default="1,2,3,4,5",
                        help="comma-separated list of lens indices to run")
    parser.add_argument("--sae_steps", type=int, default=6000)
    parser.add_argument("--sae_seqs", type=int, default=4096)
    parser.add_argument("--lens4_seqs", type=int, default=8192)
    args = parser.parse_args()

    selected = set(int(s) for s in args.lenses.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_path = load_seed_model(args.seed, device=device)
    out_dir = f"experiments/seeds/{args.seed}/lens_outputs"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Running lenses {sorted(selected)} on {ckpt_path}", flush=True)

    if 1 in selected:
        print("=== Lens 1: weight decomposition ===", flush=True)
        t0 = time.time()
        run_lens1(model, out_dir)
        print(f"Lens 1 done in {time.time()-t0:.1f}s", flush=True)
    if 5 in selected:
        print("=== Lens 5: compiled spec ===", flush=True)
        t0 = time.time()
        run_lens5(model, out_dir)
        print(f"Lens 5 done in {time.time()-t0:.1f}s", flush=True)
    if 4 in selected:
        print("=== Lens 4: polyhedral ===", flush=True)
        t0 = time.time()
        run_lens4(model, out_dir, n_seqs=args.lens4_seqs, device=device)
        print(f"Lens 4 done in {time.time()-t0:.1f}s", flush=True)
    if 3 in selected:
        print("=== Lens 3: causal ===", flush=True)
        t0 = time.time()
        run_lens3(model, out_dir, n_batches=4, batch_size=256, device=device)
        print(f"Lens 3 done in {time.time()-t0:.1f}s", flush=True)
    if 2 in selected:
        print("=== Lens 2: SAEs + transcoders ===", flush=True)
        t0 = time.time()
        run_lens2(model, out_dir, expansions=(8, 32),
                  n_seqs=args.sae_seqs, n_steps=args.sae_steps, device=device)
        run_transcoders(model, out_dir, expansions=(8, 32),
                        n_seqs=args.sae_seqs, n_steps=args.sae_steps, device=device)
        print(f"Lens 2 done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
