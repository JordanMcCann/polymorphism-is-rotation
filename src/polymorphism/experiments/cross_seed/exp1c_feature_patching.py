"""Experiment 1c: Cross-seed SAE feature patching.

Question: if we ablate a task-relevant feature defined on seed 0's SAE,
does the same ablation produce a comparable downstream behavioural effect
when applied to seed N's residual stream?

Setup:
  1. Identify task-relevant features in seed 0's SAE at a layer where the
     task signal lives. Concretely, we use resid_post_1 (= resid_pre_2,
     the residual stream entering the unembed); regress each SAE feature
     against three task targets:
       - sticky-invalid (binary)
       - depth >= 4 (binary)
       - is-closer (token type, binary)
     Pick the top-K features per target by correlation.

  2. For each picked feature f, compute the ablation delta in residual-
     stream coordinates:
        delta_f(x)  =  - relu(W_enc[f] . (x - pre_bias) + b_enc[f]) * W_dec[:, f]
     This is the SAE's standard write-direction representation of feature f.

  3. Patch by adding delta_f to the residual stream at the same site, run
     forward, measure (i) loss change on the target head, (ii) target-head
     accuracy change, (iii) KL vs unpatched output.

  4. Compare:
       * Within-seed: patch on seed 0 (does the feature actually matter?)
       * Cross-seed: SAME delta function applied to seed N
       * Control: ablate a random feature (matched activation magnitude)
       * Control: ablate at a random direction with matched norm

Interpretation: if the cross-seed effect matches the within-seed effect,
SAE features defined on seed 0 are valid intervention handles on seed N
(i.e., functional transfer despite coordinate-frame drift). If not, SAE
features only label seed-0-specific internal structure.
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
    load_sae,
    load_seed_model,
    make_eval_batch,
)

SEEDS = [0, 1, 2, 3, 4]


@torch.no_grad()
def feature_activations(sae, acts_BTD: torch.Tensor) -> torch.Tensor:
    """sae.encode but on a [B, T, d] tensor; returns [B, T, d_feat]."""
    B, T, D = acts_BTD.shape
    feats = sae.encode(acts_BTD.reshape(-1, D))
    return feats.reshape(B, T, -1)


@torch.no_grad()
def find_task_relevant_features(sae, acts_BTD: torch.Tensor,
                                  batch: dict, device: str) -> dict:
    """For each task target (sticky-invalid, depth>=4, is-closer), find the
    top-K SAE features by point-biserial correlation with that target.

    Returns dict: target -> list of (feature_idx, correlation, mean_act_on_pos).
    """
    feats = feature_activations(sae, acts_BTD)        # [B, T, F]
    B, T, Fdim = feats.shape
    mask = batch["mask"].to(device).bool()

    # Targets (binary), flattened to [N_valid]
    valid_flat = batch["valid"].to(device).reshape(-1)[mask.reshape(-1)]
    depth_flat = batch["depth"].to(device).reshape(-1)[mask.reshape(-1)]
    tok_flat   = batch["tok"].to(device).reshape(-1)[mask.reshape(-1)]
    closer_set = torch.tensor([4, 6, 8], device=device)     # ')', ']', '}'
    targets = {
        "sticky_invalid": valid_flat.float(),
        "depth_ge_4":     (depth_flat >= 4).float(),
        "is_closer":      torch.isin(tok_flat, closer_set).float(),
    }
    feats_flat = feats.reshape(-1, Fdim)[mask.reshape(-1)]   # [N_valid, F]

    out = {}
    for tname, t in targets.items():
        t_c = t - t.mean()
        t_n = t_c.norm()
        f_c = feats_flat - feats_flat.mean(dim=0, keepdim=True)
        f_n = f_c.norm(dim=0).clamp(min=1e-9)
        corr = (f_c * t_c.unsqueeze(1)).sum(dim=0) / (t_n.clamp(min=1e-9) * f_n)
        # Sort by absolute correlation, take top-K
        K = 5
        topk_idx = corr.abs().topk(K).indices
        out[tname] = [
            {"feature": int(i),
             "corr": float(corr[i].item()),
             "active_rate": float((feats_flat[:, i] > 0).float().mean().item()),
             "mean_act_when_target_1": float(feats_flat[t > 0, i].mean().item()
                                              if (t > 0).any() else 0.0),
             "mean_act_when_target_0": float(feats_flat[t == 0, i].mean().item()
                                              if (t == 0).any() else 0.0)}
            for i in topk_idx.tolist()
        ]
    return out


def make_ablate_delta(sae, feature_idx: int, alpha: float = 1.0):
    """Return a delta_fn(x) -> [B, T, d] that ablates feature `feature_idx`,
    scaled by `alpha` (1.0 = full ablate; >1 = over-suppress; -1 = double up).

    Uses the SAE's linear feature representation: subtract the feature's
    decoder write * the feature's pre-relu encoder activation (clipped to
    relu post-activation, i.e. only ablate where the feature actually fires).
    """
    @torch.no_grad()
    def delta_fn(x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        # encoder
        h = (flat - sae.pre_bias) @ sae.W_enc.t() + sae.b_enc
        a = F.relu(h)[:, feature_idx]                     # [N]
        # write direction
        w = sae.W_dec[:, feature_idx]                     # [D]
        delta_flat = -alpha * a.unsqueeze(1) * w.unsqueeze(0)     # [N, D]
        return delta_flat.reshape(B, T, D)
    return delta_fn


def make_random_direction_delta(sae, feature_idx: int, seed: int = 0,
                                  alpha: float = 1.0):
    """Like ablate_delta but with a random unit direction (matched magnitude
    via the same feature's activation magnitudes).
    """
    g = torch.Generator(device=sae.W_dec.device).manual_seed(seed)
    rand_dir = torch.randn(sae.W_dec.shape[0], generator=g, device=sae.W_dec.device)
    rand_dir = rand_dir / rand_dir.norm()
    @torch.no_grad()
    def delta_fn(x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        h = (flat - sae.pre_bias) @ sae.W_enc.t() + sae.b_enc
        a = F.relu(h)[:, feature_idx]                     # [N]
        delta_flat = -alpha * a.unsqueeze(1) * rand_dir.unsqueeze(0)
        return delta_flat.reshape(B, T, D)
    return delta_fn


@torch.no_grad()
def measure_patch_effect(model, batch, delta_fn, site, device,
                          ref_logits) -> dict:
    """Apply delta_fn at site, measure KL vs ref_logits and per-head loss/acc."""
    tok = batch["tok"].to(device)
    mask = batch["mask"].to(device)
    new_out = forward_with_residual_edit(model, tok, site, delta_fn)
    kl = kl_against_reference(ref_logits, new_out, mask)
    # per-head loss/acc on the new outputs
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


def run(device: str = "cuda", batch_size: int = 512, site: str = "resid_mid_1",
         expansion: int = 32, alpha: float = 4.0,
         out_path: str = "experiments/cross_seed/exp1c_feature_patching.json"):
    print("[exp1c] loading models", flush=True)
    models = {s: load_seed_model(s, device=device)[0] for s in SEEDS}
    batch = make_eval_batch(batch_size=batch_size, length_range=(2, 48))
    print(f"[exp1c] site={site} x{expansion} alpha={alpha}", flush=True)
    sae = load_sae(0, site, expansion, device=device)

    # Identify task-relevant features on seed 0 acts
    acts0 = collect_acts(models[0], [site], batch, device=device)[site]
    relevant = find_task_relevant_features(sae, acts0, batch, device)
    print(f"[exp1c] task-relevant features (on seed 0 at {site}):", flush=True)
    for tname, lst in relevant.items():
        for d in lst[:3]:
            print(f"   target={tname}: f={d['feature']} corr={d['corr']:.3f} "
                  f"act_rate={d['active_rate']:.3f}", flush=True)

    # Baseline: unmodified outputs per seed
    ref_logits = {}
    base_metrics = {}
    for s in SEEDS:
        ref_logits[s] = models[s](batch["tok"].to(device))
        base_metrics[s] = evaluate_outputs(models[s], batch, device=device)

    # For each (target, top-1 feature) pair, run the patch on every seed
    results = {"site": site, "expansion": expansion, "alpha": alpha,
                "relevant_features": relevant,
                "baseline": base_metrics,
                "patches": {}}

    for tname, lst in relevant.items():
        for f_info in lst[:3]:                       # top 3 per target
            f = f_info["feature"]
            key = f"{tname}_f{f}"
            results["patches"][key] = {"target": tname, "feature": f,
                                        "corr": f_info["corr"],
                                        "active_rate": f_info["active_rate"]}
            ablate = make_ablate_delta(sae, f, alpha=alpha)
            rand = make_random_direction_delta(sae, f,
                                                seed=hash((f, tname)) & 0xFFFFFFFF,
                                                alpha=alpha)
            for s in SEEDS:
                eff_abl = measure_patch_effect(
                    models[s], batch, ablate, site, device, ref_logits[s])
                eff_rand = measure_patch_effect(
                    models[s], batch, rand, site, device, ref_logits[s])
                results["patches"][key][f"seed{s}_ablate"] = eff_abl
                results["patches"][key][f"seed{s}_random_dir"] = eff_rand
            # Concise log line
            kl0 = results["patches"][key]["seed0_ablate"]["kl_mean"]
            kl1 = results["patches"][key]["seed1_ablate"]["kl_mean"]
            kls = [results["patches"][key][f"seed{s}_ablate"]["kl_mean"]
                   for s in SEEDS if s != 0]
            r0 = results["patches"][key]["seed0_random_dir"]["kl_mean"]
            target_head = ("valid" if tname == "sticky_invalid"
                           else "depth" if tname.startswith("depth") else "tok")
            base_acc = base_metrics[0][f"acc_{target_head}"]
            patched_acc0 = results["patches"][key]["seed0_ablate"][f"acc_{target_head}"]
            patched_acc1 = results["patches"][key]["seed1_ablate"][f"acc_{target_head}"]
            print(f"  patch {key}: KL(s0)={kl0:.4f} KL(s1)={kl1:.4f} "
                  f"meanKL(s1..4)={np.mean(kls):.4f} rand(s0)={r0:.4f} | "
                  f"acc{target_head}: base={base_acc:.3f} "
                  f"s0->{patched_acc0:.3f} s1->{patched_acc1:.3f}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[exp1c] wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--site", default="resid_mid_1")
    p.add_argument("--expansion", type=int, default=32)
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--out", default="experiments/cross_seed/exp1c_feature_patching.json")
    args = p.parse_args()
    run(device=args.device, batch_size=args.batch_size, site=args.site,
        expansion=args.expansion, alpha=args.alpha, out_path=args.out)
