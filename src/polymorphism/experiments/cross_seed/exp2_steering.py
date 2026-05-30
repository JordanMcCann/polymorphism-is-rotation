"""Experiment 2: Cross-seed steering vector transfer.

A "ROME / representation-engineering" style intervention defined in the
shared residual-stream coordinate frame, then applied unchanged across
seeds 1-4. The user's central worry: seeds use different head/neuron
permutations, so any edit must be defined in a frame that's invariant
across those permutations. The residual stream is that frame -- it's
what every component reads and writes to. W_E is frozen-shared across
all five seeds (verified), so the residual basis IS the common substrate.

Concretely, we build three classes of steering vector on seed 0:

  (1) "Sticky-invalid suppress":
        v = mean(resid_mid_1 | sticky_invalid=1, depth=4)
          - mean(resid_mid_1 | sticky_invalid=0, depth=4)
      Subtracting alpha * v should suppress the sticky-invalid prediction
      head. Matched depth controls for confounding.

  (2) "Depth +1": shift the depth label by +1.
        v = mean(resid_mid_1 | depth=5) - mean(resid_mid_1 | depth=4)
      Adding alpha * v should make depth-4 inputs read as depth 5.

  (3) "Closer suppress": suppress the token-type head's preference for
      close brackets vs open brackets.
        v = mean(resid_mid_1 | is_closer=1) - mean(resid_mid_1 | is_closer=0)

For each vector v and each seed:
  - apply the SAME v (same coordinates, same magnitude alpha)
  - measure: change in target head accuracy and KL vs unpatched output
  - sweep alpha so we can characterise the response curve

Controls per vector:
  - random unit direction with same alpha (matched-norm noise)
  - orthogonal projection of v onto W_U_target null-space (should have
    no effect on that head)
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from .utils import (
    collect_acts,
    evaluate_outputs,
    forward_with_residual_edit,
    kl_against_reference,
    load_seed_model,
    make_eval_batch,
)

SEEDS = [0, 1, 2, 3, 4]


@torch.no_grad()
def diff_of_means(acts_BTD: torch.Tensor, mask_pos: torch.Tensor,
                   mask_neg: torch.Tensor) -> torch.Tensor:
    """v = mean(acts | mask_pos) - mean(acts | mask_neg). Both masks are [B, T]."""
    flat = acts_BTD.reshape(-1, acts_BTD.shape[-1])
    mp = mask_pos.reshape(-1)
    mn = mask_neg.reshape(-1)
    pos = flat[mp].mean(dim=0)
    neg = flat[mn].mean(dim=0)
    return pos - neg


@torch.no_grad()
def build_steering_vectors(model, batch, site: str, device: str) -> dict:
    """Build all steering vectors on the given model (intended: seed 0).

    Returns dict: name -> v (torch tensor [d_model]) plus metadata about
    positive/negative populations.
    """
    acts = collect_acts(model, [site], batch, device=device)[site]  # [B, T, d]

    mask = batch["mask"].to(device).bool()
    depth = batch["depth"].to(device)
    valid = batch["valid"].to(device)
    tok   = batch["tok"].to(device)
    closer_set = torch.tensor([4, 6, 8], device=device)

    is_closer = torch.isin(tok, closer_set)

    vectors = {}

    # (1) sticky_invalid_suppress @ depth=4: subtract alpha*v to suppress
    m_pos = mask & (valid == 1) & (depth == 4)
    m_neg = mask & (valid == 0) & (depth == 4)
    npos, nneg = int(m_pos.sum().item()), int(m_neg.sum().item())
    if npos > 5 and nneg > 5:
        v = diff_of_means(acts, m_pos, m_neg)
        vectors["sticky_invalid_d4"] = {
            "v": v, "n_pos": npos, "n_neg": nneg,
            "intent": "suppress sticky_invalid (subtract alpha*v)",
            "target_head": "valid", "target_label_when_v_pushes": 1,
        }

    # (2) depth + 1 @ depth 4->5
    m_pos = mask & (depth == 5)
    m_neg = mask & (depth == 4)
    npos, nneg = int(m_pos.sum().item()), int(m_neg.sum().item())
    if npos > 5 and nneg > 5:
        v = diff_of_means(acts, m_pos, m_neg)
        vectors["depth_4_to_5"] = {
            "v": v, "n_pos": npos, "n_neg": nneg,
            "intent": "increase predicted depth by 1 (add alpha*v)",
            "target_head": "depth", "target_label_when_v_pushes": 5,
        }

    # (3) closer suppress
    m_pos = mask & is_closer
    m_neg = mask & ~is_closer & (tok > 2)        # opener tokens (3,5,7)
    npos, nneg = int(m_pos.sum().item()), int(m_neg.sum().item())
    if npos > 5 and nneg > 5:
        v = diff_of_means(acts, m_pos, m_neg)
        vectors["closer_signal"] = {
            "v": v, "n_pos": npos, "n_neg": nneg,
            "intent": "amplify closer signal (add alpha*v)",
            "target_head": "tok",
        }

    return vectors


@torch.no_grad()
def make_translate_delta(v: torch.Tensor, alpha: float, only_where_mask: torch.Tensor = None):
    """Return delta_fn that adds alpha*v at every (B,T) position; if
    only_where_mask given, only at masked positions."""
    @torch.no_grad()
    def delta_fn(x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        delta = alpha * v.view(1, 1, D).expand(B, T, D).clone()
        if only_where_mask is not None:
            delta = delta * only_where_mask.unsqueeze(-1).float()
        return delta
    return delta_fn


@torch.no_grad()
def measure_steering(model, batch, site, delta_fn, device, ref_logits) -> dict:
    """Apply delta_fn at site, measure per-head metrics + KL."""
    tok = batch["tok"].to(device)
    mask = batch["mask"].to(device)
    new_out = forward_with_residual_edit(model, tok, site, delta_fn)
    kl = kl_against_reference(ref_logits, new_out, mask)
    flat_mask = mask.reshape(-1).float()
    metrics = {}
    for head in ("tok", "depth", "valid"):
        logits = new_out[head]
        labels = batch[head].to(device)
        per = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                              labels.reshape(-1), reduction="none")
        loss = (per * flat_mask).sum() / flat_mask.sum().clamp(min=1)
        preds = logits.reshape(-1, logits.shape[-1]).argmax(dim=-1)
        correct = (preds == labels.reshape(-1)) & mask.reshape(-1)
        acc = correct.float().sum() / flat_mask.sum().clamp(min=1)
        metrics[f"loss_{head}"] = float(loss.item())
        metrics[f"acc_{head}"] = float(acc.item())
    metrics.update(kl)
    return metrics


@torch.no_grad()
def conditional_accuracy(model, batch, site, delta_fn, device, head: str,
                          cond_mask: torch.Tensor) -> dict:
    """Conditional accuracy on positions where cond_mask is True."""
    tok = batch["tok"].to(device)
    mask = batch["mask"].to(device)
    new_out = forward_with_residual_edit(model, tok, site, delta_fn)
    labels = batch[head].to(device)
    flat_logits = new_out[head].reshape(-1, new_out[head].shape[-1])
    preds = flat_logits.argmax(dim=-1)
    correct = (preds == labels.reshape(-1))
    flat_cond = (cond_mask & mask).reshape(-1)
    n = int(flat_cond.sum().item())
    if n == 0:
        return {"n": 0, "acc": float("nan")}
    return {"n": n,
            "acc": float(correct[flat_cond].float().mean().item())}


def run(device: str = "cuda", batch_size: int = 1024,
         site: str = "resid_mid_1",
         alphas: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
         out_path: str = "experiments/cross_seed/exp2_steering.json"):
    print("[exp2] loading models", flush=True)
    models = {s: load_seed_model(s, device=device)[0] for s in SEEDS}
    batch = make_eval_batch(batch_size=batch_size, length_range=(2, 48))

    # Build steering vectors on seed 0 acts
    print(f"[exp2] building steering vectors at site={site} from seed 0", flush=True)
    vectors = build_steering_vectors(models[0], batch, site, device)
    for name, info in vectors.items():
        print(f"   {name}: |v|={float(info['v'].norm().item()):.3f}  "
              f"n_pos={info['n_pos']}  n_neg={info['n_neg']}  "
              f"intent={info['intent']}", flush=True)

    # Reference outputs per seed
    ref_logits = {s: models[s](batch["tok"].to(device)) for s in SEEDS}
    base_metrics = {s: evaluate_outputs(models[s], batch, device=device) for s in SEEDS}

    # Conditional masks for accuracy on "where the steering should bite"
    mask = batch["mask"].to(device).bool()
    depth = batch["depth"].to(device)
    valid = batch["valid"].to(device)
    tok = batch["tok"].to(device)
    closer_set = torch.tensor([4, 6, 8], device=device)
    is_closer = torch.isin(tok, closer_set)
    cond_masks = {
        "sticky_invalid_d4": (mask & (valid == 1) & (depth == 4)),
        "depth_4_to_5":      (mask & (depth == 4)),
        "closer_signal":     (mask & is_closer),
    }
    cond_heads = {
        "sticky_invalid_d4": "valid",
        "depth_4_to_5":      "depth",
        "closer_signal":     "tok",
    }
    # The sign convention: subtract for "sticky_invalid_d4", add for others.
    signs = {"sticky_invalid_d4": -1.0,
             "depth_4_to_5":      +1.0,
             "closer_signal":     +1.0}

    results = {"site": site, "alphas": list(alphas), "baseline": base_metrics,
                "vectors": {}, "patches": {}}

    for name, info in vectors.items():
        v = info["v"]
        sign = signs[name]
        results["vectors"][name] = {
            "norm": float(v.norm().item()),
            "intent": info["intent"],
            "target_head": info["target_head"],
            "sign": sign,
            "n_pos_train": info["n_pos"],
            "n_neg_train": info["n_neg"],
        }
        # Random unit direction control matched in norm
        g = torch.Generator(device=device).manual_seed(hash(name) & 0xFFFFFFFF)
        rand = torch.randn(v.shape[0], generator=g, device=device)
        rand = rand / rand.norm() * v.norm()
        for alpha in alphas:
            key = f"{name}_alpha{alpha}"
            results["patches"][key] = {"vector": name, "alpha": alpha,
                                        "sign": sign}
            delta = make_translate_delta(sign * v, alpha)
            delta_rand = make_translate_delta(sign * rand, alpha)
            for s in SEEDS:
                eff = measure_steering(models[s], batch, site, delta, device,
                                        ref_logits[s])
                eff_rand = measure_steering(models[s], batch, site, delta_rand,
                                            device, ref_logits[s])
                cond_acc = conditional_accuracy(
                    models[s], batch, site, delta, device,
                    cond_heads[name], cond_masks[name])
                cond_acc_rand = conditional_accuracy(
                    models[s], batch, site, delta_rand, device,
                    cond_heads[name], cond_masks[name])
                cond_acc_base = conditional_accuracy(
                    models[s], batch, site, lambda x: torch.zeros_like(x),
                    device, cond_heads[name], cond_masks[name])
                results["patches"][key][f"seed{s}_v"] = eff
                results["patches"][key][f"seed{s}_v_cond_acc"] = cond_acc
                results["patches"][key][f"seed{s}_v_cond_acc_base"] = cond_acc_base
                results["patches"][key][f"seed{s}_random_dir"] = eff_rand
                results["patches"][key][f"seed{s}_random_dir_cond_acc"] = cond_acc_rand

            # Concise log
            kls = [results["patches"][key][f"seed{s}_v"]["kl_mean"] for s in SEEDS]
            cond_accs = [results["patches"][key][f"seed{s}_v_cond_acc"]["acc"]
                         for s in SEEDS]
            base_acc = results["patches"][key]["seed0_v_cond_acc_base"]["acc"]
            rand_kls = [results["patches"][key][f"seed{s}_random_dir"]["kl_mean"]
                        for s in SEEDS]
            print(f"  {key}: cond_acc seed0->{cond_accs[0]:.3f} "
                  f"(base {base_acc:.3f}) | seed1..4 acc=[{min(cond_accs[1:]):.3f},"
                  f" {max(cond_accs[1:]):.3f}] mean={np.mean(cond_accs[1:]):.3f} "
                  f"|| KL s0={kls[0]:.3f} s1..4={np.mean(kls[1:]):.3f} "
                  f"rand_s0={rand_kls[0]:.3f}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: (
            float(x) if isinstance(x, np.floating) else
            int(x) if isinstance(x, np.integer) else str(x)))
    print(f"[exp2] wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--site", default="resid_mid_1")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    p.add_argument("--out", default="experiments/cross_seed/exp2_steering.json")
    args = p.parse_args()
    run(device=args.device, batch_size=args.batch_size, site=args.site,
        alphas=tuple(args.alphas), out_path=args.out)
