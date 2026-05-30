"""EXP 4 — cross-checkpoint rotation within a single training run.

§9.4 prediction 1 (preregistered, before this experiment):
  "Cross-checkpoint SAE transfer should improve substantially under a
   learned orthogonal rotation. ... The strength of the recovery measures
   how much of the inter-checkpoint drift is rotation versus genuine
   function change."

Two variants:

(a) Local Dyck-3, seed 0:
    - SAE_30k: train an SAE on activations from the step-30000 checkpoint
    - acts_58k: activations from the step-58000 checkpoint (= the analysis ckpt
      used throughout the paper)
    - Apply SAE_30k naively to acts_58k -> measure EV
    - Fit a one-batch Procrustes R between (acts_30k, acts_58k)
    - Apply SAE_30k to (acts_58k @ R) -> measure EV
    - Improvement under R measures rotation-vs-function drift

(b) Pythia-70m-seed1:
    - Use revisions step3000 and step143000 (downloaded by download_pythia.py)
    - Train an SAE on step3000 activations on one site
    - Apply to step143000 activations naively and after Procrustes
    - Compare improvement factor to (a) and to the cross-seed EXP 1 factor

Outputs:
  experiments/cross_checkpoint/local_dyck.json
  experiments/cross_checkpoint/pythia.json
  experiments/cross_checkpoint/figure_cross_ckpt.png
"""

from __future__ import annotations

import gc
import json
import os
import time

import torch

from ...analysis.lens2_saes import SAE, SAEConfig, train_sae
from ...model import Config, Transformer, make_model
from ..cross_seed.utils import SITES as DYCK_SITES
from ..cross_seed.utils import make_eval_batch, resolve_site
from ..scale.common import (
    CorpusConfig,
    best_orthogonal,
    collect_residual_activations,
    flatten_acts,
    load_pythia,
    procrustes_metrics,
    stream_text_chunks,
    tokenize_corpus,
)

OUT_DIR = "experiments/cross_checkpoint"


# ===================== variant (a): local Dyck-3 =====================

def load_dyck_ckpt(seed: int, step: int, device: str = "cuda") -> Transformer:
    """Load a specific Dyck-3 checkpoint by step number."""
    path = f"experiments/seeds/{seed}/checkpoints/ckpt_{step:07d}.pt"
    state = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = state.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items()
                     if k in Config.__dataclass_fields__})
    m = make_model(cfg)
    m.load_state_dict(state["model_state"])
    return m.to(device).eval()


@torch.no_grad()
def collect_dyck_activations_flat(model: Transformer, batch: dict,
                                    device: str) -> dict[str, torch.Tensor]:
    """Collect Dyck-3 residual activations at the 7 standard sites."""
    tok = batch["tok"].to(device)
    mask = batch["mask"].to(device)
    _, cache = model(tok, return_internals=True)
    blocks = cache["blocks"]
    out = {}
    for s in DYCK_SITES:
        a = resolve_site(blocks, s)
        flat = a.reshape(-1, a.shape[-1])
        out[s] = flat[mask.reshape(-1)]
    return out


def variant_a_local_dyck(seed: int = 0, step_train_sae: int = 30000,
                            step_target: int = 58000,
                            n_sae_steps: int = 3000,
                            device: str = "cuda") -> dict:
    """Run variant (a). Returns dict per site with naive vs post-rotation EV."""
    print(f"[exp4-a] loading checkpoints (seed={seed}, steps={step_train_sae} "
          f"and {step_target}) ...", flush=True)
    m_train = load_dyck_ckpt(seed, step_train_sae, device=device)
    m_target = load_dyck_ckpt(seed, step_target, device=device)
    batch = make_eval_batch(batch_size=1024, length_range=(2, 48), seed=2026)

    print("[exp4-a] collecting activations ...", flush=True)
    acts_train = collect_dyck_activations_flat(m_train, batch, device)
    acts_target = collect_dyck_activations_flat(m_target, batch, device)

    results = {"seed": seed, "step_train_sae": step_train_sae,
                "step_target": step_target,
                "per_site": {}}

    for site in DYCK_SITES:
        a_tr = acts_train[site].float()
        a_tg = acts_target[site].float()
        print(f"[exp4-a]   site={site} ({a_tr.shape[0]} samples, d={a_tr.shape[1]})",
              flush=True)
        # Train SAE on a_tr
        cfg = SAEConfig(d_in=a_tr.shape[1], expansion=8, n_steps=n_sae_steps,
                        batch_size=4096, lr=5e-4, l1_coef=1e-3, seed=0)
        sae_res = train_sae(a_tr, cfg, device=device, verbose=False)
        sae_state = sae_res["state"]
        sae_cfg_dict = sae_res["config"]
        print(f"[exp4-a]     SAE EV(self/train) = {sae_res['explained_var']:.4f}, "
              f"L0={sae_res['sparsity_l0']:.2f}", flush=True)
        # Naive: apply SAE to a_tg
        sae = SAE(SAEConfig(**sae_cfg_dict)).to(device)
        sae.load_state_dict(sae_state); sae.eval()
        with torch.no_grad():
            recon, _ = sae(a_tg.to(device))
            err = (recon - a_tg.to(device)) ** 2
            mse_raw = float(err.mean().item())
            var = float(a_tg.var().item())
            ev_raw = 1.0 - mse_raw / max(var, 1e-12)
        # Procrustes
        R = best_orthogonal(a_tg, a_tr)
        mean_tg_rot = (a_tg.mean(0) @ R)
        shift = a_tr.mean(0) - mean_tg_rot
        a_tg_rot = a_tg @ R + shift
        with torch.no_grad():
            recon_r, _ = sae(a_tg_rot.to(device))
            err_r = (recon_r - a_tg_rot.to(device)) ** 2
            mse_rot = float(err_r.mean().item())
            ev_rot = 1.0 - mse_rot / max(var, 1e-12)
        rot_metrics = procrustes_metrics(a_tg, a_tr, R)
        # Improvement factor
        rec_improvement = ev_rot - ev_raw
        results["per_site"][site] = {
            "sae_self_train_EV": float(sae_res["explained_var"]),
            "naive_cross_ckpt_EV": float(ev_raw),
            "post_rotation_EV": float(ev_rot),
            "improvement_under_rotation": float(rec_improvement),
            "rotation_audit": rot_metrics,
            "n_samples": int(a_tr.shape[0]),
            "d_model": int(a_tr.shape[1]),
        }
        print(f"[exp4-a]     naive_EV={ev_raw:.4f}, rot_EV={ev_rot:.4f}, "
              f"improvement={rec_improvement:.4f}, "
              f"||R-I||={rot_metrics['frob_R_minus_I']:.3f}", flush=True)
    del m_train, m_target
    torch.cuda.empty_cache(); gc.collect()
    return results


# ===================== variant (b): Pythia =====================

def variant_b_pythia(model_id: str = "EleutherAI/pythia-70m-seed1",
                       rev_train: str = "step3000",
                       rev_target: str = "step143000",
                       layer_idx: int = 3,
                       n_sequences: int = 256, seq_len: int = 256,
                       n_sae_steps: int = 4000,
                       device: str = "cuda") -> dict:
    """Variant (b). Train SAE on early-ckpt activations, apply to late-ckpt
    activations naively and post-Procrustes.

    Defaults: pythia-70m-seed1 at step3000 (very early) vs step143000 (final).
    layer_idx=3 (mid-stack of 6 layers).
    """
    from transformers import AutoTokenizer
    print("[exp4-b] loading tokenizer (shared across revisions) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"[exp4-b] preparing corpus ({n_sequences} seq) ...", flush=True)
    cfg = CorpusConfig(n_sequences=n_sequences, seq_len=seq_len, seed=2026)
    texts = stream_text_chunks(cfg)
    tokens = tokenize_corpus(texts, tok, cfg)

    site = f"layer{layer_idx}_resid_post"
    print(f"[exp4-b] target site: {site}", flush=True)

    # Train ckpt activations
    print(f"[exp4-b] loading {model_id} rev={rev_train} ...", flush=True)
    m_tr, _ = load_pythia(model_id, revision=rev_train, dtype="bf16",
                           device=device)
    acts_tr = collect_residual_activations(m_tr, tokens, batch_size=8,
                                              device=device)
    flat_tr = flatten_acts(acts_tr)[site].float()
    del m_tr
    torch.cuda.empty_cache(); gc.collect()

    # Target ckpt activations
    print(f"[exp4-b] loading {model_id} rev={rev_target} ...", flush=True)
    m_tg, _ = load_pythia(model_id, revision=rev_target, dtype="bf16",
                           device=device)
    acts_tg = collect_residual_activations(m_tg, tokens, batch_size=8,
                                              device=device)
    flat_tg = flatten_acts(acts_tg)[site].float()
    del m_tg
    torch.cuda.empty_cache(); gc.collect()

    print(f"[exp4-b] training SAE on rev={rev_train} acts (d={flat_tr.shape[1]}) ...",
          flush=True)
    sae_cfg = SAEConfig(d_in=flat_tr.shape[1], expansion=8, n_steps=n_sae_steps,
                          batch_size=4096, lr=5e-4, l1_coef=1e-3, seed=0)
    sae_res = train_sae(flat_tr, sae_cfg, device=device, verbose=False)
    print(f"[exp4-b]   SAE self EV={sae_res['explained_var']:.4f}, "
          f"L0={sae_res['sparsity_l0']:.2f}", flush=True)

    # Naive
    sae = SAE(SAEConfig(**sae_res["config"])).to(device)
    sae.load_state_dict(sae_res["state"]); sae.eval()
    with torch.no_grad():
        r, _ = sae(flat_tg.to(device))
        ev_raw = float(1.0 - ((r - flat_tg.to(device)) ** 2).mean().item()
                        / max(flat_tg.var().item(), 1e-12))
    # Procrustes
    R = best_orthogonal(flat_tg, flat_tr)
    mean_tg_rot = (flat_tg.mean(0) @ R)
    shift = flat_tr.mean(0) - mean_tg_rot
    flat_tg_rot = flat_tg @ R + shift
    with torch.no_grad():
        r_rot, _ = sae(flat_tg_rot.to(device))
        ev_rot = float(1.0 - ((r_rot - flat_tg_rot.to(device)) ** 2).mean().item()
                        / max(flat_tg_rot.var().item(), 1e-12))
    rot_metrics = procrustes_metrics(flat_tg, flat_tr, R)

    return {
        "model_id": model_id, "rev_train_sae": rev_train,
        "rev_target": rev_target, "layer_idx": layer_idx, "site": site,
        "sae_self_train_EV": float(sae_res["explained_var"]),
        "naive_cross_ckpt_EV": float(ev_raw),
        "post_rotation_EV": float(ev_rot),
        "improvement_under_rotation": float(ev_rot - ev_raw),
        "rotation_audit": rot_metrics,
        "n_samples": int(flat_tr.shape[0]), "d_model": int(flat_tr.shape[1]),
    }


def run_all(device: str = "cuda", out_dir: str = OUT_DIR,
             variants: tuple[str, ...] = ("a", "b"),
             n_sae_steps_a: int = 3000, n_sae_steps_b: int = 4000) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    if "a" in variants:
        print("[exp4] variant (a) — local Dyck-3 cross-checkpoint", flush=True)
        t0 = time.time()
        res_a = variant_a_local_dyck(device=device, n_sae_steps=n_sae_steps_a)
        res_a["wall_sec"] = time.time() - t0
        with open(os.path.join(out_dir, "local_dyck.json"), "w") as f:
            json.dump(res_a, f, indent=2, default=str)
        results["variant_a"] = res_a
    if "b" in variants:
        print("[exp4] variant (b) — Pythia cross-revision", flush=True)
        t0 = time.time()
        res_b = variant_b_pythia(device=device, n_sae_steps=n_sae_steps_b)
        res_b["wall_sec"] = time.time() - t0
        with open(os.path.join(out_dir, "pythia.json"), "w") as f:
            json.dump(res_b, f, indent=2, default=str)
        results["variant_b"] = res_b
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--variants", default="ab")
    args = p.parse_args()
    variants = tuple(args.variants)
    run_all(device=args.device, variants=variants)
