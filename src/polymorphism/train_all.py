"""Train all 5 seeds sequentially (or in series; GPU is small).

Writes per-seed logs to logs/train_seed{N}.json and checkpoints to
experiments/seeds/{N}/checkpoints/."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Allow `python -m polymorphism.train_all` and `python src/polymorphism/train_all.py` both:
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from polymorphism.model import Config
    from polymorphism.train import TrainCfg, train_one_seed
else:
    from .model import Config
    from .train import TrainCfg, train_one_seed

# Tensors that make up the shared "I/O basis" (token embed + the three unembed
# heads). train.py loads exactly these keys via --shared_io_init_path.
SHARED_IO_KEYS = ("W_E", "W_U_tok", "W_U_depth", "W_U_valid")


def extract_shared_io(source_seed: int, out_path: str | None) -> None:
    """Pull the shared I/O tensors from a trained seed's best checkpoint and
    save them as a dict consumable by train.py's --shared_io_init_path.

    Used by `replicate run-full` to derive Cohort A's frozen I/O basis from
    the freshly trained seed 0 before training seeds 1-4 against it.
    """
    if not out_path:
        raise SystemExit("--extract_shared_io requires --out PATH")
    import torch

    from polymorphism.run_lenses import load_seed_model
    model, ckpt = load_seed_model(source_seed, device="cpu", which="best")
    shared = {k: getattr(model, k).detach().cpu().clone() for k in SHARED_IO_KEYS}
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(shared, out_path)
    print(f"[extract] shared I/O {list(shared)} from seed {source_seed} "
          f"({ckpt}) -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--n_steps", type=int, default=30_000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--ckpt_every", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--extra_after_convergence", type=int, default=3000)
    parser.add_argument("--shared_io_init_path", type=str, default=None,
                        help="Optional path to shared I/O init for cross-seed coordination")
    parser.add_argument("--freeze_shared_io", action="store_true",
                        help="Freeze shared I/O tensors during training")
    parser.add_argument("--extract_shared_io", action="store_true",
                        help="Extract the shared I/O tensors from a trained seed, "
                             "write them to --out, and exit (no training)")
    parser.add_argument("--source_seed", type=int, default=0,
                        help="Seed whose trained I/O basis to extract "
                             "(used with --extract_shared_io)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for --extract_shared_io")
    args = parser.parse_args()

    if args.extract_shared_io:
        extract_shared_io(args.source_seed, args.out)
        return

    seeds = [int(s) for s in args.seeds.split(",")]
    out_root = "experiments/seeds"
    os.makedirs("logs", exist_ok=True)
    summaries = []

    for seed in seeds:
        out_dir = os.path.join(out_root, str(seed))
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join("logs", f"train_seed{seed}.json")
        cfg = Config()
        tc = TrainCfg(
            seed=seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            ckpt_every=args.ckpt_every,
            eval_every=args.eval_every,
            extra_steps_after_first_pass=args.extra_after_convergence,
            out_dir=out_dir,
            shared_io_init_path=args.shared_io_init_path,
            freeze_shared_io=args.freeze_shared_io,
        )
        print(f"=== Training seed {seed} ===", flush=True)
        t0 = time.time()
        summary = train_one_seed(cfg, tc, log_path=log_path)
        elapsed = time.time() - t0
        summary["elapsed_min"] = elapsed / 60
        summaries.append(summary)
        print(f"=== Seed {seed} done in {elapsed/60:.1f} min ===", flush=True)
        with open(os.path.join("logs", "train_summary.json"), "w") as f:
            json.dump(summaries, f, indent=2, default=str)

    print("ALL SEEDS DONE", flush=True)


if __name__ == "__main__":
    main()
