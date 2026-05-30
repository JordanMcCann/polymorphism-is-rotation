"""Shared infrastructure for EXP 1, EXP 4, EXP 5 at Pythia scale.

This module is the single source for:
  - Pythia model loading (bf16 to fit RTX 2060 12GB easily; pythia-70m is
    ~166 MB in bf16 so we have generous VRAM headroom)
  - A streaming text corpus (The Pile, with NeelNanda/pile-10k fallback)
  - Generic activation collection (hidden_states[i] = residual after block i)
  - Generic Procrustes (orthogonal Frobenius fit) — reproduces the helper
    from polymorphism/experiments/cross_seed/exp1d_rotation_audit.py at arbitrary d_in
  - SAE adapter (re-uses polymorphism.analysis.lens2_saes.SAE; only d_in changes)
  - Activation cache layout helpers

Site naming convention for transformer-stack models:
    layer{L}_resid_post   — output of block L's residual stream
                            (= hidden_states[L+1] in HF's convention,
                             since hidden_states[0] is the embedding)
    layer0_resid_pre      — input to block 0 (= hidden_states[0])

This file imports torch heavyweight stuff only inside functions so that
test collection stays fast.
"""

from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

# ----------------- model loading -----------------

def load_pythia(model_id: str, revision: str | None = None,
                dtype: str = "bf16", device: str = "cuda"):
    """Load a Pythia model + tokenizer for inference (no grad, eval mode).

    dtype: 'bf16' (default; fits everything in 1GB even for 160m) or 'fp32'.

    Returns (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    kwargs = {"torch_dtype": torch_dtype}
    if revision is not None:
        kwargs["revision"] = revision
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def model_d_model(model) -> int:
    """Return d_model for a HF transformer (works for GPT-NeoX/Pythia)."""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    if hasattr(model.config, "d_model"):
        return model.config.d_model
    raise AttributeError("Cannot find d_model on model.config")


def model_n_layers(model) -> int:
    """Return number of layers."""
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    if hasattr(model.config, "n_layer"):
        return model.config.n_layer
    raise AttributeError("Cannot find n_layers on model.config")


def site_list(n_layers: int) -> list[str]:
    """Standard residual-stream site list for a transformer with n_layers blocks."""
    sites = ["layer0_resid_pre"]
    for L in range(n_layers):
        sites.append(f"layer{L}_resid_post")
    return sites


# ----------------- streaming corpus -----------------

@dataclass
class CorpusConfig:
    n_sequences: int = 1024
    seq_len: int = 512
    seed: int = 2026
    # The activation cache key is derived from this dataset name, so keep it
    # stable. `monology/pile-uncopyrighted` is zstd-compressed and needs the
    # optional `zstandard` package to stream; without it, stream_text_chunks
    # falls back to NeelNanda/pile-10k -- the corpus the published artifacts
    # were built with. The fallback is expected, not an error.
    dataset: str = "monology/pile-uncopyrighted"
    split: str = "train"


def stream_text_chunks(cfg: CorpusConfig) -> list[str]:
    """Return a fixed list of `cfg.n_sequences` text chunks, deterministic in cfg.seed.

    Tries `cfg.dataset` first; falls back to NeelNanda/pile-10k if access fails.
    """
    from datasets import load_dataset
    sources = [cfg.dataset]
    if cfg.dataset != "NeelNanda/pile-10k":
        sources.append("NeelNanda/pile-10k")

    last_error = None
    for src in sources:
        try:
            print(f"[corpus] trying {src} (streaming=True) ...", flush=True)
            if src == "NeelNanda/pile-10k":
                ds = load_dataset(src, split="train", streaming=False)
                # already a small, in-memory dataset
                rng = np.random.default_rng(cfg.seed)
                idx = rng.choice(len(ds), size=min(cfg.n_sequences * 4, len(ds)),
                                  replace=False)
                texts = [ds[int(i)]["text"] for i in idx]
            else:
                ds = load_dataset(src, split=cfg.split, streaming=True)
                texts = []
                for ex in ds.shuffle(seed=cfg.seed, buffer_size=10_000):
                    t = ex.get("text") or ex.get("content") or ""
                    if len(t) > 200:  # need long-ish chunks
                        texts.append(t)
                    if len(texts) >= cfg.n_sequences * 2:
                        break
            print(f"[corpus]   got {len(texts)} candidate texts from {src}", flush=True)
            return texts[: cfg.n_sequences * 2]
        except Exception as e:
            last_error = e
            print(f"[corpus]   {src} unavailable ({type(e).__name__}: {str(e)[:160]}); "
                  f"trying the next source -- expected without the optional `zstandard` package",
                  flush=True)
            continue
    raise RuntimeError(f"All corpus sources failed; last: {last_error}")


def tokenize_corpus(texts: list[str], tokenizer, cfg: CorpusConfig) -> torch.Tensor:
    """Tokenize and chunk-to-fixed-length. Returns int64 [n_sequences, seq_len]."""
    out = []
    for t in texts:
        if len(out) >= cfg.n_sequences:
            break
        ids = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=cfg.seq_len, padding=False)["input_ids"][0]
        if ids.numel() >= cfg.seq_len:
            out.append(ids[: cfg.seq_len])
    if len(out) < cfg.n_sequences:
        raise RuntimeError(f"Got only {len(out)} long-enough chunks; "
                            f"requested {cfg.n_sequences}")
    return torch.stack(out, dim=0)


def get_fixed_token_batch(tokenizer, cfg: CorpusConfig | None = None) -> torch.Tensor:
    """Convenience: one-shot Pile-tokens batch for EXP 1, EXP 4, EXP 5.

    Deterministic given cfg.seed and the tokenizer's vocab.
    """
    cfg = cfg or CorpusConfig()
    texts = stream_text_chunks(cfg)
    return tokenize_corpus(texts, tokenizer, cfg)


# ----------------- activation collection -----------------

@torch.no_grad()
def collect_residual_activations(model, tokens: torch.Tensor, batch_size: int = 32,
                                  device: str = "cuda") -> dict[str, torch.Tensor]:
    """Run `tokens` through `model` and collect hidden_states at every residual site.

    Returns dict site_name -> tensor of shape [N_seq, T, d_model] on CPU, in fp32
    (cheaper to cast once than to repeatedly mix dtypes downstream).
    """
    n_layers = model_n_layers(model)
    sites = site_list(n_layers)
    chunks = {s: [] for s in sites}

    N = tokens.shape[0]
    for i in range(0, N, batch_size):
        batch = tokens[i : i + batch_size].to(device)
        out = model(batch, output_hidden_states=True, use_cache=False)
        # hidden_states is a tuple of length n_layers + 1
        # hidden_states[0] = embeddings (= input to block 0 = resid_pre)
        # hidden_states[L+1] = output of block L = resid_post of L
        hs = out.hidden_states
        for j, s in enumerate(sites):
            chunks[s].append(hs[j].detach().cpu().float())
    return {s: torch.cat(chunks[s], dim=0) for s in sites}


def flatten_acts(acts_by_site: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flatten [N, T, d] -> [N*T, d] for every site. Useful for SAE/Procrustes."""
    return {s: a.reshape(-1, a.shape[-1]) for s, a in acts_by_site.items()}


# ----------------- activation cache -----------------

def cache_key(model_id: str, revision: str | None, corpus_cfg: CorpusConfig,
              site: str) -> str:
    """Filename-safe key uniquely identifying an activation cache file."""
    rev = revision or "latest"
    bits = f"{model_id}|{rev}|{corpus_cfg.dataset}|{corpus_cfg.n_sequences}|" \
           f"{corpus_cfg.seq_len}|{corpus_cfg.seed}|{site}"
    h = hashlib.sha1(bits.encode()).hexdigest()[:12]
    safe_id = model_id.replace("/", "__")
    return f"{safe_id}__{rev}__{site}__{h}"


def cache_path(cache_dir: str, model_id: str, revision: str | None,
               corpus_cfg: CorpusConfig, site: str, suffix: str = ".npy") -> str:
    key = cache_key(model_id, revision, corpus_cfg, site)
    return os.path.join(cache_dir, key + suffix)


def save_cached_acts(cache_dir: str, model_id: str, revision: str | None,
                      corpus_cfg: CorpusConfig,
                      acts_by_site: dict[str, torch.Tensor]) -> dict[str, str]:
    """Persist a per-site activation dict to .npy files keyed deterministically."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for s, t in acts_by_site.items():
        p = cache_path(cache_dir, model_id, revision, corpus_cfg, s)
        np.save(p, t.numpy().astype(np.float16))
        paths[s] = p
    return paths


def load_cached_acts(cache_dir: str, model_id: str, revision: str | None,
                      corpus_cfg: CorpusConfig,
                      sites: list[str]) -> dict[str, torch.Tensor] | None:
    """Load cached activations if all `sites` are present. Returns None if any missing."""
    out = {}
    for s in sites:
        p = cache_path(cache_dir, model_id, revision, corpus_cfg, s)
        if not os.path.exists(p):
            return None
        out[s] = torch.from_numpy(np.load(p)).float()
    return out


# ----------------- generic Procrustes -----------------

@torch.no_grad()
def best_orthogonal(acts_src: torch.Tensor, acts_tgt: torch.Tensor) -> torch.Tensor:
    """Procrustes: orthogonal R minimising ||acts_src @ R - acts_tgt||_F.

    Both inputs are [N, d] (un-centred). The function centres them internally
    via subtract-mean (so the returned R is the rotation alone, not a rotation-
    plus-translation). Caller is responsible for translation if desired.
    """
    a_src = (acts_src - acts_src.mean(0, keepdim=True)).double()
    a_tgt = (acts_tgt - acts_tgt.mean(0, keepdim=True)).double()
    M = a_src.T @ a_tgt
    U, _, Vt = torch.linalg.svd(M)
    return (U @ Vt).to(acts_src.dtype)


@torch.no_grad()
def procrustes_metrics(acts_src: torch.Tensor, acts_tgt: torch.Tensor,
                        R: torch.Tensor | None = None) -> dict:
    """Compute the standard rotation-audit metrics for (src -> tgt).

    Returns:
      raw_EV          — explained variance of identity (src ≈ tgt)
      rot_EV          — explained variance after rotation
      frob_R_minus_I  — || R - I ||_F
      op_norm_R       — operator (spectral) norm of R (should be 1.0)
    """
    if R is None:
        R = best_orthogonal(acts_src, acts_tgt)
    src_c = (acts_src - acts_src.mean(0, keepdim=True)).float()
    tgt_c = (acts_tgt - acts_tgt.mean(0, keepdim=True)).float()
    var_t = float(tgt_c.pow(2).mean().item())
    err_raw = (src_c - tgt_c).pow(2).mean().item()
    err_rot = (src_c @ R - tgt_c).pow(2).mean().item()
    I = torch.eye(R.shape[0], device=R.device, dtype=R.dtype)
    return {
        "raw_EV": float(1.0 - err_raw / max(var_t, 1e-12)),
        "rot_EV": float(1.0 - err_rot / max(var_t, 1e-12)),
        "frob_R_minus_I": float((R - I).pow(2).sum().sqrt().item()),
        "op_norm_R": float(torch.linalg.norm(R, ord=2).item()),
        "d_model": int(R.shape[0]),
    }


# ----------------- SAE adapter -----------------

def train_sae_on(acts: torch.Tensor, expansion: int = 8, n_steps: int = 6000,
                  l1_coef: float = 1e-3, lr: float = 5e-4,
                  batch_size: int = 4096, seed: int = 0,
                  device: str = "cuda") -> dict:
    """Thin wrapper over polymorphism.analysis.lens2_saes.train_sae that auto-fills d_in.

    Returns the standard train_sae result dict.
    """
    from ...analysis.lens2_saes import SAEConfig, train_sae
    cfg = SAEConfig(d_in=acts.shape[1], expansion=expansion, n_steps=n_steps,
                    l1_coef=l1_coef, lr=lr, batch_size=batch_size, seed=seed)
    return train_sae(acts, cfg, device=device, verbose=False)


def sae_apply(state: dict, sae_cfg: dict, acts: torch.Tensor,
              device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """Forward an SAE state on activations. Returns (recon, features) on CPU."""
    from ...analysis.lens2_saes import SAE, SAEConfig
    cfg = SAEConfig(**sae_cfg)
    sae = SAE(cfg).to(device)
    sae.load_state_dict(state)
    sae.eval()
    with torch.no_grad():
        a = acts.to(device).float()
        recon, feats = sae(a)
    return recon.cpu(), feats.cpu()


def sae_metrics(sae_state: dict, sae_cfg: dict, acts: torch.Tensor,
                 device: str = "cuda") -> dict:
    """Reconstruction MSE, explained variance, L0 sparsity for SAE applied to acts."""
    recon, feats = sae_apply(sae_state, sae_cfg, acts, device=device)
    err = recon - acts
    recon_mse = float((err ** 2).mean().item())
    var_x = float(acts.var().item())
    explained_var = float(1.0 - recon_mse / max(var_x, 1e-12))
    sparsity_l0 = float((feats > 0).float().sum(dim=1).mean().item())
    return {"recon_mse": recon_mse, "explained_var": explained_var,
            "sparsity_l0": sparsity_l0, "n": int(acts.shape[0])}


# ----------------- decoder cosine -----------------

def decoder_cosine_histogram(state_a: dict, state_b: dict,
                              threshold: float = 0.5) -> dict:
    """For each feature in A, find max cosine with any feature in B.

    Returns dict with fraction_stable (frac > threshold), mean_max_cos, and
    the full histogram bins for plotting.
    """
    Wa = state_a["W_dec"].cpu().numpy() if isinstance(state_a["W_dec"], torch.Tensor) \
         else state_a["W_dec"]
    Wb = state_b["W_dec"].cpu().numpy() if isinstance(state_b["W_dec"], torch.Tensor) \
         else state_b["W_dec"]
    A = Wa / (np.linalg.norm(Wa, axis=0, keepdims=True) + 1e-9)
    B = Wb / (np.linalg.norm(Wb, axis=0, keepdims=True) + 1e-9)
    cos = A.T @ B            # [d_feat_a, d_feat_b]
    max_per_a = cos.max(axis=1)
    bins = np.linspace(-1.0, 1.0, 21)
    hist, _ = np.histogram(max_per_a, bins=bins)
    return {
        "fraction_stable": float((max_per_a > threshold).mean()),
        "mean_max_cos": float(max_per_a.mean()),
        "median_max_cos": float(np.median(max_per_a)),
        "p05_max_cos": float(np.percentile(max_per_a, 5)),
        "p95_max_cos": float(np.percentile(max_per_a, 95)),
        "histogram_counts": hist.tolist(),
        "histogram_bins": bins.tolist(),
        "n_features_a": int(Wa.shape[1]),
        "n_features_b": int(Wb.shape[1]),
        "threshold": threshold,
    }


# ----------------- smoke test -----------------

def smoke():
    """End-to-end smoke test of the infrastructure: load pythia-70m, run a tiny
    batch, fit one Procrustes, train one mini-SAE.

    Runs in < 60s on RTX 2060 if Pythia is already cached.
    """
    print("[smoke] loading pythia-70m bf16 ...", flush=True)
    model, tok = load_pythia("EleutherAI/pythia-70m", dtype="bf16")
    print(f"[smoke]   d_model={model_d_model(model)} n_layers={model_n_layers(model)}",
          flush=True)
    cfg = CorpusConfig(n_sequences=16, seq_len=128, seed=42)
    print("[smoke] streaming a small corpus ...", flush=True)
    texts = stream_text_chunks(cfg)
    tokens = tokenize_corpus(texts, tok, cfg)
    print(f"[smoke]   tokens shape: {tokens.shape}", flush=True)
    print("[smoke] collecting activations ...", flush=True)
    acts = collect_residual_activations(model, tokens, batch_size=4)
    for s, a in acts.items():
        print(f"[smoke]   {s}: {tuple(a.shape)}", flush=True)
    # Procrustes on a synthetic seed-N analog: rotate by a known R
    site0 = "layer0_resid_post"
    flat = flatten_acts({site0: acts[site0]})[site0]
    R_true = torch.linalg.qr(torch.randn(flat.shape[1], flat.shape[1]))[0]
    rotated = flat @ R_true
    R_fit = best_orthogonal(rotated, flat)
    err = (rotated @ R_fit - flat).pow(2).mean().sqrt().item()
    print(f"[smoke] Procrustes recovery error: {err:.2e}", flush=True)
    # Mini SAE
    print("[smoke] training mini-SAE (x4, 200 steps) ...", flush=True)
    res = train_sae_on(flat, expansion=4, n_steps=200, batch_size=512,
                        l1_coef=1e-3, lr=5e-4)
    print(f"[smoke]   SAE EV: {res['explained_var']:.4f}, "
          f"L0: {res['sparsity_l0']:.1f}", flush=True)
    print("[smoke] OK", flush=True)


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        smoke()
    else:
        print("Usage: python -m polymorphism.experiments.scale.common --smoke")
