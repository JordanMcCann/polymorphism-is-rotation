"""Lens 2 -- Sparse Autoencoders (SAEs) and transcoders.

For every internal "site" of the model:
   residual_pre   (residual stream entering layer L)
   resid_mid      (after attention, before MLP)
   resid_post     (residual stream leaving layer L)
   attn_out       (the head outputs themselves)
   mlp_pre        (the MLP pre-ReLU activations)
   mlp_post       (the MLP post-ReLU activations)

we train sparse autoencoders at expansion factors 8x, 32x, 128x. We also
train a transcoder per layer mapping MLP-input -> MLP-output (the MLP
viewed as a single function approximated by sparse features).

Implementation:
   - SAE: encoder W_enc [d_feat, d_in], bias_enc [d_feat], decoder W_dec
     [d_in, d_feat]. ReLU activation. L1 penalty on activations.
   - Pre-encoder bias `pre_bias` per Anthropic recipe.
   - Loss: reconstruction MSE + l1_coef * |features|_1.
   - Optimiser: Adam.

A "feature" produced by an SAE is interpreted by:
   1. Activation pattern over a fixed evaluation set.
   2. The direction in residual space it writes (column of W_dec).
   3. Cross-correlation with the next site's features (transcoder).

Cross-SAE-seed stability is measured by training two SAEs with different
seeds at the same site and matching features via cosine similarity.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..model import Config, Transformer
from ..task import TaskConfig, sample_batch

SITES = [
    "resid_pre_0", "resid_mid_0", "resid_post_0",
    "attn_out_0", "mlp_pre_0",   "mlp_post_0",
    "resid_pre_1", "resid_mid_1", "resid_post_1",
    "attn_out_1", "mlp_pre_1",   "mlp_post_1",
    "resid_pre_2",            # = resid_post_1 of the last block, before ln_f
]


def site_dim(cfg: Config, site: str) -> int:
    if "mlp_pre" in site or "mlp_post" in site:
        return cfg.d_mlp
    return cfg.d_model


def collect_activations(
    model: Transformer, sites: list[str], n_seqs: int = 4096, batch: int = 128,
    device: str = "cuda", seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Run the model on n_seqs random inputs, collect activations at each site.

    Returns a dict site -> tensor of shape [n_seqs * T_valid, dim].
    """
    model.eval()
    rng = np.random.default_rng(seed)
    task_cfg = TaskConfig(n_ctx=model.cfg.n_ctx)
    n_batches = (n_seqs + batch - 1) // batch
    per_site: dict[str, list[torch.Tensor]] = {s: [] for s in sites}

    with torch.no_grad():
        for _ in range(n_batches):
            b = sample_batch(batch, task_cfg, rng, length_range=(2, 48))
            tok = b["tok"].to(device)
            mask = b["mask"].to(device)
            _, cache = model(tok, return_internals=True)
            blocks = cache["blocks"]
            for s in sites:
                tensor = _resolve_site(s, blocks, cache, model)
                # Flatten to [B*T, dim] then filter mask
                flat = tensor.reshape(-1, tensor.shape[-1])
                m = mask.reshape(-1)
                per_site[s].append(flat[m].cpu())

    return {s: torch.cat(per_site[s], dim=0) for s in sites}


def _resolve_site(s: str, blocks: list, cache, model: Transformer) -> torch.Tensor:
    """Map a site name to the right activation tensor."""
    if s == "resid_pre_0":
        return blocks[0]["resid_pre"]
    if s == "resid_mid_0":
        return blocks[0]["resid_mid"]
    if s == "resid_post_0":
        return blocks[0]["resid_post"]
    if s == "attn_out_0":
        return blocks[0]["attn_out"]
    if s == "mlp_pre_0":
        return blocks[0]["mlp_pre"]
    if s == "mlp_post_0":
        return blocks[0]["mlp_post"]
    if s == "resid_pre_1":
        return blocks[1]["resid_pre"]
    if s == "resid_mid_1":
        return blocks[1]["resid_mid"]
    if s == "resid_post_1":
        return blocks[1]["resid_post"]
    if s == "attn_out_1":
        return blocks[1]["attn_out"]
    if s == "mlp_pre_1":
        return blocks[1]["mlp_pre"]
    if s == "mlp_post_1":
        return blocks[1]["mlp_post"]
    if s == "resid_pre_2":
        # = resid_post of last block (input to ln_f)
        return blocks[-1]["resid_post"]
    raise KeyError(s)


@dataclass
class SAEConfig:
    d_in: int = 64
    expansion: int = 8
    l1_coef: float = 1e-3
    n_steps: int = 8000
    batch_size: int = 4096
    lr: float = 5e-4
    seed: int = 0


class SAE(nn.Module):
    """ReLU-activation SAE with pre-encoder bias and L1 sparsity penalty."""

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        d_feat = cfg.d_in * cfg.expansion
        self.d_feat = d_feat
        # Pre-encoder bias
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        # Encoder
        W = torch.randn(d_feat, cfg.d_in) / (cfg.d_in ** 0.5)
        self.W_enc = nn.Parameter(W)
        self.b_enc = nn.Parameter(torch.zeros(d_feat))
        # Decoder (rows unit-norm)
        W_dec = torch.randn(cfg.d_in, d_feat) / (d_feat ** 0.5)
        # normalize decoder columns to unit length
        W_dec = W_dec / (W_dec.norm(dim=0, keepdim=True) + 1e-9)
        self.W_dec = nn.Parameter(W_dec)

    def encode(self, x):
        # x: [N, d_in]
        h = (x - self.pre_bias) @ self.W_enc.t() + self.b_enc
        return F.relu(h)

    def decode(self, features):
        return features @ self.W_dec.t() + self.pre_bias

    def forward(self, x):
        feats = self.encode(x)
        recon = self.decode(feats)
        return recon, feats

    @torch.no_grad()
    def normalise_decoder(self):
        """Project decoder columns to unit norm; absorb scale into encoder."""
        norms = self.W_dec.norm(dim=0, keepdim=True) + 1e-9
        self.W_dec.data /= norms
        self.W_enc.data *= norms.squeeze(0).unsqueeze(1)
        self.b_enc.data *= norms.squeeze(0)


def train_sae(activations: torch.Tensor, cfg: SAEConfig, device="cuda",
              verbose: bool = False) -> dict:
    """Train one SAE on the given activation matrix [N, d_in].

    Returns dict with the trained SAE state, final reconstruction error,
    sparsity, dead-feature rate, and feature norms."""
    torch.manual_seed(cfg.seed)
    sae = SAE(cfg).to(device)
    optim = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    N = activations.shape[0]
    activations = activations.to(device).float()
    # Track which features have ever fired
    ever_fired = torch.zeros(sae.d_feat, dtype=torch.bool, device=device)
    sample_metrics = {"loss": [], "recon": [], "sparsity": []}

    it = range(cfg.n_steps)
    for step in (tqdm(it, desc=f"SAE d_in={cfg.d_in} exp={cfg.expansion}",
                      file=sys.stdout) if verbose else it):
        idx = torch.randint(0, N, (cfg.batch_size,), device=device)
        x = activations[idx]
        recon, feats = sae(x)
        recon_loss = ((recon - x) ** 2).mean()
        l1 = feats.abs().sum(dim=1).mean()
        loss = recon_loss + cfg.l1_coef * l1
        optim.zero_grad(); loss.backward()
        # Decoder column normalisation gradient projection (Anthropic recipe)
        with torch.no_grad():
            d = sae.W_dec.grad
            proj = (d * sae.W_dec).sum(dim=0, keepdim=True) * sae.W_dec
            sae.W_dec.grad = d - proj
        optim.step()
        with torch.no_grad():
            sae.normalise_decoder()
            ever_fired |= (feats > 0).any(dim=0)
        if step % max(1, cfg.n_steps // 10) == 0:
            sample_metrics["loss"].append(loss.item())
            sample_metrics["recon"].append(recon_loss.item())
            sample_metrics["sparsity"].append(feats.gt(0).float().sum(dim=1).mean().item())

    # Final stats on the full activation set
    with torch.no_grad():
        feats_all = []
        for i in range(0, N, 8192):
            x = activations[i : i + 8192]
            feats_all.append(sae.encode(x).cpu())
        feats_all = torch.cat(feats_all, dim=0)
        recon_full = sae.decode(feats_all.to(device))
        recon_mse = ((recon_full - activations) ** 2).mean().item()
        var_x = activations.var().item()
        explained_var = 1 - recon_mse / max(var_x, 1e-12)
        sparsity = feats_all.gt(0).float().sum(dim=1).mean().item()
        dead = (~ever_fired).float().mean().item()
        feature_freq = feats_all.gt(0).float().mean(dim=0)

    return {
        "state": {k: v.cpu().detach() for k, v in sae.state_dict().items()},
        "config": cfg.__dict__,
        "recon_mse": recon_mse,
        "explained_var": explained_var,
        "sparsity_l0": sparsity,
        "dead_feature_rate": dead,
        "feature_freq": feature_freq.tolist(),
        "metrics_history": sample_metrics,
    }


def cross_seed_feature_stability(state_a: dict, state_b: dict,
                                  threshold: float = 0.5) -> dict:
    """For each feature in SAE-a, find the most-similar feature in SAE-b by
    decoder-direction cosine similarity. Returns the fraction with cos >
    threshold, plus the mean of max-cosines.

    A high `fraction_stable` indicates the SAE has learned features that
    transfer across seeds — strong evidence the features are not artifacts
    of optimisation noise but are picking up genuine signal in the residual.
    """
    W_dec_a = state_a["W_dec"]
    W_dec_b = state_b["W_dec"]
    if isinstance(W_dec_a, torch.Tensor):
        W_dec_a = W_dec_a.cpu().numpy()
        W_dec_b = W_dec_b.cpu().numpy()
    A = W_dec_a / (np.linalg.norm(W_dec_a, axis=0, keepdims=True) + 1e-9)
    B = W_dec_b / (np.linalg.norm(W_dec_b, axis=0, keepdims=True) + 1e-9)
    cos = A.T @ B
    max_per_a = cos.max(axis=1)
    return {
        "fraction_stable": float((max_per_a > threshold).mean()),
        "mean_max_cos":    float(max_per_a.mean()),
        "n_features_a":    int(W_dec_a.shape[1]),
        "n_features_b":    int(W_dec_b.shape[1]),
        "threshold":       threshold,
    }


def run_lens2(model: Transformer, out_dir: str, sites: list[str] | None = None,
              expansions: tuple[int, ...] = (8, 32, 128), n_seqs: int = 4096,
              device: str = "cuda", n_steps: int = 6000) -> dict:
    """Train SAEs at every site at every expansion factor. Saves each to disk."""
    os.makedirs(out_dir, exist_ok=True)
    if sites is None:
        # default: residual streams and MLP sites
        sites = [
            "resid_pre_0", "resid_mid_0", "resid_post_0",
            "mlp_pre_0", "mlp_post_0",
            "resid_pre_1", "resid_mid_1", "resid_post_1",
            "mlp_pre_1", "mlp_post_1",
            "resid_pre_2",
        ]

    print(f"[Lens 2] collecting activations from {n_seqs} sequences...", flush=True)
    acts = collect_activations(model, sites, n_seqs=n_seqs, device=device)
    print(f"[Lens 2] activations per site: {next(iter(acts.values())).shape[0]} rows",
          flush=True)

    summary = {}
    for s in sites:
        for exp in expansions:
            cfg = SAEConfig(
                d_in=acts[s].shape[1], expansion=exp,
                n_steps=n_steps, l1_coef=1e-3, seed=0,
            )
            print(f"[Lens 2] training SAE at site={s} expansion={exp}x ...", flush=True)
            res = train_sae(acts[s], cfg, device=device, verbose=False)
            # Save weights
            save_path = os.path.join(out_dir, f"sae_{s}_x{exp}.pt")
            torch.save({
                "state": res["state"], "config": res["config"],
                "metrics": {k: v for k, v in res.items()
                            if k not in ("state", "feature_freq", "metrics_history")},
                "feature_freq": res["feature_freq"],
            }, save_path)
            summary.setdefault(s, {})[f"x{exp}"] = {
                "recon_mse": res["recon_mse"],
                "explained_var": res["explained_var"],
                "sparsity_l0": res["sparsity_l0"],
                "dead_feature_rate": res["dead_feature_rate"],
                "saved": save_path,
            }
            print(f"   recon_mse={res['recon_mse']:.5f} "
                  f"explained_var={res['explained_var']:.4f} "
                  f"L0={res['sparsity_l0']:.1f} dead={res['dead_feature_rate']:.2%}",
                  flush=True)
        # also train a second SAE with a different seed for stability check
        cfg = SAEConfig(d_in=acts[s].shape[1], expansion=expansions[0],
                        n_steps=n_steps, l1_coef=1e-3, seed=1)
        res2 = train_sae(acts[s], cfg, device=device, verbose=False)
        torch.save({"state": res2["state"], "config": res2["config"]},
                   os.path.join(out_dir, f"sae_{s}_x{expansions[0]}_seed1.pt"))
        summary[s]["replicate_seed1_recon"] = res2["recon_mse"]

    with open(os.path.join(out_dir, "lens2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_transcoders(model: Transformer, out_dir: str, expansions: tuple[int, ...] = (8, 32),
                    n_seqs: int = 4096, device: str = "cuda", n_steps: int = 6000) -> dict:
    """Train transcoders: MLP-input -> MLP-output, layer by layer.

    A transcoder is structurally the same as an SAE but with a different
    decoder target (the *output* of the MLP, not the input). It gives a
    sparse-feature account of the MLP as a function.
    """
    os.makedirs(out_dir, exist_ok=True)
    sites_in_out = [
        ("ln2_out_0", "mlp_out_0"),
        ("ln2_out_1", "mlp_out_1"),
    ]
    # Collect both inputs and outputs together
    model.eval()
    rng = np.random.default_rng(0)
    task_cfg = TaskConfig(n_ctx=model.cfg.n_ctx)
    n_batches = (n_seqs + 127) // 128
    pairs = [([], []) for _ in sites_in_out]
    with torch.no_grad():
        for _ in range(n_batches):
            b = sample_batch(128, task_cfg, rng, length_range=(2, 48))
            tok = b["tok"].to(device); mask = b["mask"].to(device)
            _, cache = model(tok, return_internals=True)
            for i, (s_in, s_out) in enumerate(sites_in_out):
                L = int(s_in.split("_")[-1])
                a_in = cache["blocks"][L]["ln2_out"]
                a_out = cache["blocks"][L]["mlp_out"]
                flat_in = a_in.reshape(-1, a_in.shape[-1])
                flat_out = a_out.reshape(-1, a_out.shape[-1])
                m = mask.reshape(-1)
                pairs[i][0].append(flat_in[m].cpu())
                pairs[i][1].append(flat_out[m].cpu())

    summary = {}
    for i, (s_in, s_out) in enumerate(sites_in_out):
        X = torch.cat(pairs[i][0], dim=0)
        Y = torch.cat(pairs[i][1], dim=0)
        for exp in expansions:
            tc_cfg = SAEConfig(d_in=X.shape[1], expansion=exp,
                               n_steps=n_steps, l1_coef=1e-3, seed=0)
            res = _train_transcoder(X, Y, tc_cfg, device=device)
            save_path = os.path.join(out_dir, f"transcoder_layer{i}_x{exp}.pt")
            torch.save({"state": res["state"], "config": res["config"],
                        "feature_freq": res.get("feature_freq", [])}, save_path)
            summary.setdefault(f"layer{i}", {})[f"x{exp}"] = {
                "recon_mse": res["recon_mse"],
                "explained_var": res["explained_var"],
                "sparsity_l0": res["sparsity_l0"],
                "saved": save_path,
            }
    with open(os.path.join(out_dir, "transcoder_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _train_transcoder(X: torch.Tensor, Y: torch.Tensor, cfg: SAEConfig,
                       device: str = "cuda") -> dict:
    torch.manual_seed(cfg.seed)
    sae = SAE(cfg).to(device)
    # Add a separate output projection for Y target (decoder reused)
    Y_proj = nn.Linear(sae.d_feat, Y.shape[1], bias=True).to(device)
    optim = torch.optim.Adam(list(sae.parameters()) + list(Y_proj.parameters()),
                              lr=cfg.lr)
    X = X.to(device).float(); Y = Y.to(device).float()
    N = X.shape[0]
    ever_fired = torch.zeros(sae.d_feat, dtype=torch.bool, device=device)
    for step in range(cfg.n_steps):
        idx = torch.randint(0, N, (cfg.batch_size,), device=device)
        x = X[idx]; y = Y[idx]
        feats = sae.encode(x)
        y_hat = Y_proj(feats)
        recon = ((y_hat - y) ** 2).mean()
        l1 = feats.abs().sum(dim=1).mean()
        loss = recon + cfg.l1_coef * l1
        optim.zero_grad(); loss.backward(); optim.step()
        with torch.no_grad():
            sae.normalise_decoder()
            ever_fired |= (feats > 0).any(dim=0)

    with torch.no_grad():
        feats_all = []
        for i in range(0, N, 8192):
            feats_all.append(sae.encode(X[i:i+8192]).cpu())
        feats_all = torch.cat(feats_all, dim=0)
        y_hat = Y_proj(feats_all.to(device))
        recon_mse = ((y_hat - Y) ** 2).mean().item()
        var_y = Y.var().item()
        explained_var = 1 - recon_mse / max(var_y, 1e-12)
        sparsity = feats_all.gt(0).float().sum(dim=1).mean().item()

    state = {**{k: v.cpu().detach() for k, v in sae.state_dict().items()},
             "Y_proj_W": Y_proj.weight.detach().cpu(),
             "Y_proj_b": Y_proj.bias.detach().cpu()}
    return {
        "state": state, "config": cfg.__dict__,
        "recon_mse": recon_mse, "explained_var": explained_var,
        "sparsity_l0": sparsity,
    }
