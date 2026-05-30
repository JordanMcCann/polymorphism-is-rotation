"""Bar 1 (Behavioral): KL(spec || trained) averaged across many inputs.

Spec: a Python implementation of the algorithm that produces target
probability distributions per output head. For our task the spec is
deterministic, so the spec's distribution is one-hot at the correct
label and zero elsewhere. To make KL finite we replace zero entries
with eps = 1e-6 and renormalise, which corresponds to assuming the
spec has a negligible-but-nonzero "I might be wrong" probability.

For 10^8 samples we use bucketed accumulation (running mean of KL),
sampling random batches from train and held-out distributions in
proportion 60/20/20. fp32 throughout (the L1->weight_decay fix makes
fp64 unnecessary). Periodic disk checkpoints make the run interruptible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from polymorphism.model import Config, Transformer
    from polymorphism.task import (
        TaskConfig,
        sample_batch,
        sample_compositional_test,
        sample_long_test,
    )
else:
    from ..model import Config, Transformer
    from ..task import (
        TaskConfig,
        sample_batch,
        sample_compositional_test,
        sample_long_test,
    )


def spec_distribution_log(batch: dict, vocab_size: int = 40, n_depth: int = 9,
                           n_valid: int = 2, eps: float = 1e-6,
                           device: str = "cuda") -> dict:
    """Return (p, log_p) per head, with eps-smoothed one-hot target."""
    out = {}
    for head, K in [("tok", vocab_size), ("depth", n_depth), ("valid", n_valid)]:
        labels = batch[head].to(device)
        p = torch.full((*labels.shape, K), eps / (K - 1),
                        dtype=torch.float32, device=device)
        p.scatter_(2, labels.unsqueeze(-1), 1.0 - eps)
        log_p = torch.log(p.clamp(min=1e-30))
        out[head] = (p, log_p)
    return out


def kl_one_hot(p: torch.Tensor, log_p: torch.Tensor, log_q: torch.Tensor,
                mask: torch.Tensor) -> tuple[float, int]:
    """KL(p || q) = sum p (log p - log q); restricted to mask positions."""
    per_pos = (p * (log_p - log_q)).sum(dim=-1)
    per_pos = per_pos * mask.float()
    return float(per_pos.sum().item()), int(mask.sum().item())


def run_bar_behavioral(model: Transformer, n_samples: int = 100_000_000,
                       batch_size: int = 2048, device: str = "cuda",
                       distributions: tuple[tuple[str, float], ...] = (
                           ("train", 0.6), ("compositional", 0.2), ("long", 0.2),
                       ),
                       seed: int = 0,
                       progress_every: int = 200,
                       checkpoint_path: str | None = None,
                       checkpoint_every: int = 50_000) -> dict:
    """Returns dict with overall KL and per-distribution breakdown."""
    model = model.to(device).eval()
    task_cfg = TaskConfig(n_ctx=model.cfg.n_ctx)
    n_batches = (n_samples + batch_size - 1) // batch_size

    # Initialize / resume from checkpoint
    start_batch = 0
    total_kl = 0.0; total_n = 0
    per_head_kl = {h: 0.0 for h in ("tok", "depth", "valid")}
    per_head_n  = {h: 0 for h in ("tok", "depth", "valid")}
    per_dist = {d: {"kl": 0.0, "n": 0} for d, _ in distributions}
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, weights_only=False)
        start_batch = state["batch"]
        total_kl = state["total_kl"]; total_n = state["total_n"]
        per_head_kl = state["per_head_kl"]; per_head_n = state["per_head_n"]
        per_dist = state["per_dist"]
        print(f"[Bar B] resuming from batch {start_batch}/{n_batches}", flush=True)

    # Distribution PMF
    dist_pmf = np.array([p for _, p in distributions])
    dist_pmf = dist_pmf / dist_pmf.sum()
    dist_rng = np.random.default_rng(seed + 1)
    data_rng = np.random.default_rng(seed + start_batch * 7 + 13)

    t0 = time.time()
    samples_per_print = batch_size * progress_every
    for bi in range(start_batch, n_batches):
        d_idx = int(dist_rng.choice(len(distributions), p=dist_pmf))
        dname = distributions[d_idx][0]
        if dname == "train":
            batch = sample_batch(batch_size, task_cfg, data_rng, length_range=(2, 48))
        elif dname == "compositional":
            batch = sample_compositional_test(batch_size, task_cfg, data_rng)
        else:
            batch = sample_long_test(batch_size, task_cfg, data_rng)
        # Move to device
        tok = batch["tok"].to(device)
        mask = batch["mask"].to(device)

        with torch.no_grad():
            out = model(tok)

        specs = spec_distribution_log(batch, vocab_size=model.cfg.vocab_size,
                                       n_depth=model.cfg.n_depth,
                                       n_valid=model.cfg.n_valid,
                                       device=device)
        for head in ("tok", "depth", "valid"):
            p, log_p = specs[head]
            log_q = F.log_softmax(out[head].float(), dim=-1)
            kl_sum, n = kl_one_hot(p, log_p, log_q, mask)
            total_kl += kl_sum; total_n += n
            per_head_kl[head] += kl_sum; per_head_n[head] += n
            per_dist[dname]["kl"] += kl_sum
            per_dist[dname]["n"] += n

        if (bi - start_batch) > 0 and (bi - start_batch) % progress_every == 0:
            done = (bi + 1) * batch_size
            elapsed = time.time() - t0
            sps = (bi + 1 - start_batch) * batch_size / max(elapsed, 1e-3)
            kl_avg = total_kl / max(total_n, 1)
            eta = (n_batches - bi - 1) * batch_size / max(sps, 1)
            print(f"[Bar B] sampled {done:.0f}/{n_samples}  "
                  f"KL/pos/head={kl_avg:.6e}  "
                  f"elapsed={elapsed:.1f}s  sps={sps:.0f}  eta={eta:.0f}s",
                  flush=True)

        if checkpoint_path and (bi - start_batch) > 0 and \
                (bi - start_batch) % checkpoint_every == 0:
            torch.save({
                "batch": bi + 1, "total_kl": total_kl, "total_n": total_n,
                "per_head_kl": per_head_kl, "per_head_n": per_head_n,
                "per_dist": per_dist,
            }, checkpoint_path)

    kl_avg = total_kl / max(total_n, 1)
    result = {
        "mean_kl": kl_avg,
        "n_samples": n_samples,
        "n_batches": n_batches,
        "per_head_kl": {h: per_head_kl[h] / max(per_head_n[h], 1) for h in per_head_kl},
        "per_distribution": {
            d: {"kl": v["kl"] / max(v["n"], 1), "n": v["n"]}
            for d, v in per_dist.items()
        },
        "passed": kl_avg < 1e-4,
        "tolerance": 1e-4,
    }
    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n_samples", type=int, default=100_000_000)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--checkpoint_every", type=int, default=50_000)
    parser.add_argument("--progress_every", type=int, default=200)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_dict = state.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                    if k in Config.__dataclass_fields__})
    from polymorphism.model import make_model
    model = make_model(cfg)
    model.load_state_dict(state["model_state"])
    result = run_bar_behavioral(model, n_samples=args.n_samples,
                                 batch_size=args.batch_size,
                                 checkpoint_path=args.checkpoint_path,
                                 checkpoint_every=args.checkpoint_every,
                                 progress_every=args.progress_every)
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
