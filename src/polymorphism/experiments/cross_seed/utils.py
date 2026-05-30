"""Shared utilities for cross-seed SAE/steering experiments.

Key assumption that justifies the whole experimental setup:
  W_E (and W_U_*) are *frozen* and shared across seeds 0-4 (Cohort A's
  shared-frozen-I/O training regime). The residual stream therefore exists in a
  common coordinate frame across seeds, even though the internal MLP/attn
  weights diverge wildly (Bar P fails at MSE ~0.5). This makes
  "apply seed-0 SAE to seed-N residual stream" and "subtract a steering
  vector defined on seed 0 from seed N's residual stream" well-defined
  operations -- the central premise being tested.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from ...analysis.lens2_saes import SAE, SAEConfig
from ...model import Config, Transformer, make_model
from ...task import TaskConfig, sample_batch

SITES = [
    "resid_pre_0", "resid_mid_0", "resid_post_0",
    "resid_pre_1", "resid_mid_1", "resid_post_1",
    "resid_pre_2",
]


def load_seed_model(seed: int, device: str = "cuda") -> tuple[Transformer, str]:
    """Load the best (or last) checkpoint for a given seed. Mirrors
    universality.load_seed but trimmed."""
    ckpts = sorted(glob.glob(f"experiments/seeds/{seed}/checkpoints/ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints for seed {seed}")
    target = ckpts[-1]
    log_path = f"logs/train_seed{seed}.json"
    if os.path.exists(log_path):
        data = json.load(open(log_path))
        evals = [r for r in data if isinstance(r, dict)
                 and "train" in r and isinstance(r.get("train"), dict)]
        if evals:
            def _key(r):
                loss_sum = 0.0
                for dist in ("train", "compositional", "long"):
                    m = r.get(dist, {})
                    for k in ("loss_tok", "loss_depth", "loss_valid"):
                        loss_sum += float(m.get(k, 0))
                return (r["eval_min_acc"], -loss_sum)
            best = max(evals, key=_key)
            step = best["step"]
            for c in ckpts:
                step_in_name = int(c.rsplit("_", 1)[1].split(".")[0])
                if step_in_name == step:
                    target = c
                    break
    state = torch.load(target, map_location=device, weights_only=False)
    cfg_dict = state.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                     if k in Config.__dataclass_fields__})
    m = make_model(cfg)
    m.load_state_dict(state["model_state"])
    return m.to(device).eval(), target


def load_sae(seed: int, site: str, expansion: int,
              device: str = "cuda") -> SAE:
    """Load a pre-trained SAE for (seed, site, expansion)."""
    path = f"experiments/seeds/{seed}/lens_outputs/sae_{site}_x{expansion}.pt"
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = blob["config"]
    cfg = SAEConfig(**cfg_dict)
    sae = SAE(cfg).to(device)
    sae.load_state_dict(blob["state"])
    sae.eval()
    return sae


def resolve_site(blocks: list, site: str) -> torch.Tensor:
    """Extract a residual-stream tensor by site name. Returns [B, T, d_model]."""
    if site == "resid_pre_0":
        return blocks[0]["resid_pre"]
    if site == "resid_mid_0":
        return blocks[0]["resid_mid"]
    if site == "resid_post_0":
        return blocks[0]["resid_post"]
    if site == "resid_pre_1":
        return blocks[1]["resid_pre"]
    if site == "resid_mid_1":
        return blocks[1]["resid_mid"]
    if site == "resid_post_1":
        return blocks[1]["resid_post"]
    if site == "resid_pre_2":
        return blocks[-1]["resid_post"]
    raise KeyError(site)


@torch.no_grad()
def collect_acts(model: Transformer, sites: list[str], batch: dict,
                  device: str = "cuda") -> dict[str, torch.Tensor]:
    """Run one batch through model, collect tensors at each site.

    Returns dict site -> [B, T, d] (full activations, NOT mask-filtered).
    Useful when you need to identify (sequence, position) pairs.
    """
    tok = batch["tok"].to(device)
    _, cache = model(tok, return_internals=True)
    blocks = cache["blocks"]
    return {s: resolve_site(blocks, s) for s in sites}


@torch.no_grad()
def collect_acts_flat(model: Transformer, sites: list[str], batch: dict,
                       device: str = "cuda") -> dict[str, torch.Tensor]:
    """Flat-masked activations: [N_valid, d]."""
    tok = batch["tok"].to(device)
    mask = batch["mask"].to(device)
    _, cache = model(tok, return_internals=True)
    out = {}
    for s in sites:
        a = resolve_site(cache["blocks"], s)
        flat = a.reshape(-1, a.shape[-1])
        out[s] = flat[mask.reshape(-1)]
    return out


def make_eval_batch(batch_size: int = 512, length_range=(2, 48),
                     seed: int = 12345) -> dict:
    """Build a single fixed evaluation batch (same across seeds).

    Crucial: seeds 1-4 must see IDENTICAL inputs for firing-pattern overlap
    measurements to be meaningful.
    """
    rng = np.random.default_rng(seed)
    task_cfg = TaskConfig(n_ctx=64)
    return sample_batch(batch_size, task_cfg, rng, length_range=length_range)


@torch.no_grad()
def evaluate_outputs(model: Transformer, batch: dict, device: str = "cuda") -> dict:
    """Compute per-head loss + accuracy for a single batch."""
    tok = batch["tok"].to(device)
    mask = batch["mask"].to(device)
    out = model(tok)
    flat_mask = mask.reshape(-1)
    metrics = {}
    for head in ("tok", "depth", "valid"):
        logits = out[head]
        labels = batch[head].to(device)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        per = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        loss = (per * flat_mask.float()).sum() / flat_mask.float().sum().clamp(min=1)
        preds = flat_logits.argmax(dim=-1)
        correct = (preds == flat_labels) & flat_mask
        acc = correct.float().sum() / flat_mask.float().sum().clamp(min=1)
        metrics[f"loss_{head}"] = float(loss.item())
        metrics[f"acc_{head}"] = float(acc.item())
    return metrics


@torch.no_grad()
def kl_against_reference(ref_logits: dict, new_logits: dict,
                          mask: torch.Tensor) -> dict:
    """Mean KL(ref || new) per head, averaged over masked positions."""
    flat_mask = mask.reshape(-1).float()
    out = {}
    for head in ("tok", "depth", "valid"):
        logp = F.log_softmax(ref_logits[head].double(), dim=-1)
        logq = F.log_softmax(new_logits[head].double(), dim=-1)
        p = logp.exp()
        kl = (p * (logp - logq)).sum(dim=-1).reshape(-1)
        out[f"kl_{head}"] = float((kl * flat_mask).sum().item()
                                    / flat_mask.sum().clamp(min=1).item())
    out["kl_mean"] = float(np.mean([out[f"kl_{h}"] for h in ("tok", "depth", "valid")]))
    return out


@torch.no_grad()
def forward_with_residual_edit(model: Transformer, tokens: torch.Tensor,
                                 site: str, delta_fn) -> dict:
    """Run forward pass, but at the named residual-stream `site` add `delta_fn(x)`
    to the residual stream before continuing.

    `delta_fn(x)` takes the residual tensor [B, T, d_model] and returns a delta
    of the same shape (added). To replace the residual entirely, return
    (replacement - x). The added/subtracted delta propagates through the rest
    of the forward pass.

    Supported sites: resid_pre_0, resid_mid_0, resid_post_0,
                     resid_pre_1, resid_mid_1, resid_post_1, resid_pre_2.
    """
    cfg = model.cfg
    B, T = tokens.shape
    x = model.W_E[tokens] + model.W_pos[:T].unsqueeze(0)

    def _maybe_edit(x, here: str):
        if here == site:
            return x + delta_fn(x)
        return x

    # resid_pre_0
    x = _maybe_edit(x, "resid_pre_0")

    # Layer 0
    blk = model.blocks[0]
    x_norm = blk.ln1(x)
    attn_out = blk.attn(x_norm)
    x = x + attn_out
    x = _maybe_edit(x, "resid_mid_0")
    x_norm2 = blk.ln2(x)
    mlp_out = blk.mlp(x_norm2)
    x = x + mlp_out
    x = _maybe_edit(x, "resid_post_0")
    x = _maybe_edit(x, "resid_pre_1")  # alias

    # Layer 1
    blk = model.blocks[1]
    x_norm = blk.ln1(x)
    attn_out = blk.attn(x_norm)
    x = x + attn_out
    x = _maybe_edit(x, "resid_mid_1")
    x_norm2 = blk.ln2(x)
    mlp_out = blk.mlp(x_norm2)
    x = x + mlp_out
    x = _maybe_edit(x, "resid_post_1")
    x = _maybe_edit(x, "resid_pre_2")  # alias

    x_final = model.ln_f(x)
    return {
        "tok":   torch.einsum("btd,vd->btv", x_final, model.W_U_tok),
        "depth": torch.einsum("btd,vd->btv", x_final, model.W_U_depth),
        "valid": torch.einsum("btd,vd->btv", x_final, model.W_U_valid),
    }
