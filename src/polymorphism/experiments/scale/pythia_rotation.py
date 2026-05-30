"""EXP 1 — Pythia multi-seed rotation replication at scale.

Four headline panels mirroring §8 of PAPER.md, but at d_model=512 and across
9 independently-trained Pythia-70m seeds (EleutherAI/pythia-70m-seed{1..9}).

Panel A — Decoder-cosine histogram across (seed_i, seed_j) feature pairs at
          every residual site, every (i, j) pair. This is the standard SAE
          universality metric.

Panel B — Naive cross-seed encoder EV: apply seed_i's SAE to seed_j's
          activations. Predicted by the rotation hypothesis to be very low
          (probably negative) at internal sites, and positive only at the
          embedding-near sites where shared tokenizer / similar embedding
          structure helps.

Panel C — One-batch Procrustes R fit per (seed_i, seed_j, site). Measure
          ||R - I||_F (should be ≈ sqrt(2*d_model) ≈ 32.0 for d_model=512
          if seeds are essentially unrelated). Post-rotation reconstruction
          EV with seed_i's SAE applied to (seed_j activations @ R).

Panel D — Diff-of-means steering vector transfer (three regimes):
          - sentiment direction (likely mostly-rotated; predicted partial)
          - IOI/name direction (pinned by shared tokenizer; predicted clean)
          - numerical-magnitude direction (intermediate)
          Each applied across seeds, dose-response measured by KL on a
          held-out batch.

Outputs:
  experiments/scale/pythia_rotation/results.json — all four panels' numbers
  (the paper figures are rendered separately by `python -m replicate figures`)

The script is structured so any panel can be run independently:
  python -m polymorphism.experiments.scale.pythia_rotation --panels A
  python -m polymorphism.experiments.scale.pythia_rotation --panels ABCD
"""

from __future__ import annotations

import gc
import json
import os
import time

import numpy as np
import torch

from .common import (
    CorpusConfig,
    best_orthogonal,
    collect_residual_activations,
    decoder_cosine_histogram,
    flatten_acts,
    load_cached_acts,
    load_pythia,
    model_d_model,
    model_n_layers,
    procrustes_metrics,
    sae_metrics,
    save_cached_acts,
    site_list,
    stream_text_chunks,
    tokenize_corpus,
    train_sae_on,
)

# Constants for EXP 1
MODELS = [f"EleutherAI/pythia-70m-seed{i}" for i in range(1, 10)]
ANCHOR = "EleutherAI/pythia-70m-seed1"  # All cross-seed comparisons anchor here
# Both eval and SAE-training corpora deliberately re-use the panel_c_fast
# config so the per-(seed, site) activation cache populated by
# pythia_panel_c_fast.py is hit directly (saves ~15 min of re-extraction).
# 256 sequences × 256 tokens = 65k activation rows per (seed, site), which
# is comfortably enough for SAE training at d_in=512.
CORPUS_CFG = CorpusConfig(n_sequences=256, seq_len=256, seed=2026,
                          dataset="monology/pile-uncopyrighted")
CORPUS_CFG_SAE = CorpusConfig(n_sequences=256, seq_len=256, seed=2026,
                              dataset="monology/pile-uncopyrighted")
CACHE_DIR = "experiments/scale/cache"
OUT_DIR = "experiments/scale/pythia_rotation"


# -------------------- corpus prep --------------------

def prepare_token_batches(device: str = "cuda"):
    """Tokenize the corpus once using a shared tokenizer. Pythia models in the
    family share the EleutherAI tokenizer, so we use seed1's tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ANCHOR)
    # SAE training tokens (larger batch)
    print(f"[corpus] streaming SAE training corpus ({CORPUS_CFG_SAE.n_sequences} seq) ...",
          flush=True)
    texts = stream_text_chunks(CORPUS_CFG_SAE)
    tokens_sae = tokenize_corpus(texts, tok, CORPUS_CFG_SAE)
    # Evaluation tokens (smaller batch, different seed)
    eval_cfg = CorpusConfig(n_sequences=CORPUS_CFG.n_sequences,
                              seq_len=CORPUS_CFG.seq_len,
                              seed=CORPUS_CFG.seed + 1,
                              dataset=CORPUS_CFG.dataset)
    print(f"[corpus] streaming eval corpus ({eval_cfg.n_sequences} seq) ...",
          flush=True)
    texts_e = stream_text_chunks(eval_cfg)
    tokens_eval = tokenize_corpus(texts_e, tok, eval_cfg)
    print(f"[corpus] SAE tokens={tuple(tokens_sae.shape)} "
          f"eval tokens={tuple(tokens_eval.shape)}", flush=True)
    return tok, tokens_sae, tokens_eval, eval_cfg


# -------------------- collect or load activations --------------------

def get_or_extract_acts(model_id: str, tokens: torch.Tensor, corpus_cfg: CorpusConfig,
                          sites: list[str], device: str = "cuda",
                          batch_size: int = 16) -> dict[str, torch.Tensor]:
    """Try to load cached activations; if missing, extract from the model and cache.

    Returns dict site -> tensor [N, T, d] on CPU fp32.
    """
    cached = load_cached_acts(CACHE_DIR, model_id, None, corpus_cfg, sites)
    if cached is not None:
        return cached
    print(f"[acts] extracting from {model_id} ({tokens.shape[0]} seq) ...", flush=True)
    model, _ = load_pythia(model_id, dtype="bf16", device=device)
    t0 = time.time()
    acts = collect_residual_activations(model, tokens, batch_size=batch_size,
                                          device=device)
    print(f"[acts]   extracted in {time.time() - t0:.1f}s", flush=True)
    save_cached_acts(CACHE_DIR, model_id, None, corpus_cfg, acts)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return acts


# -------------------- SAE training per (model, site) --------------------

def train_saes_for_model(model_id: str, sites: list[str], tokens: torch.Tensor,
                          corpus_cfg: CorpusConfig, expansions=(8, 32),
                          n_steps: int = 4000, l1_coef: float = 1e-5,
                          device: str = "cuda") -> dict:
    """Train SAEs at every (site, expansion) for one model. Returns dict
    (site, expansion) -> sae_result dict (with state + metrics).

    Default l1_coef=1e-5 is calibrated for Pythia-scale activations whose
    typical magnitudes are O(10) per dim; l1_coef=1e-3 (as used for Dyck-3
    where activations are O(1)) over-penalises and the SAE collapses to
    near-zero feature usage.
    """
    acts = get_or_extract_acts(model_id, tokens, corpus_cfg, sites, device=device)
    # Flatten to [N*T, d_model]
    flat = flatten_acts(acts)
    saes = {}
    for site in sites:
        a = flat[site]
        for exp in expansions:
            cache_path = os.path.join(
                CACHE_DIR, "saes",
                f"sae_{model_id.replace('/', '__')}_{site}_x{exp}.pt")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            if os.path.exists(cache_path):
                blob = torch.load(cache_path, map_location="cpu",
                                    weights_only=False)
                saes[(site, exp)] = {
                    "state": blob["state"], "config": blob["config"],
                    "metrics": blob.get("metrics", {})
                }
                continue
            print(f"[sae] training {model_id} {site} x{exp} ...", flush=True)
            t0 = time.time()
            res = train_sae_on(a, expansion=exp, n_steps=n_steps,
                                 batch_size=4096, lr=5e-4, l1_coef=l1_coef,
                                 seed=0, device=device)
            metrics_dict = {k: v for k, v in res.items()
                            if k not in ("state", "feature_freq", "metrics_history")}
            torch.save({"state": res["state"], "config": res["config"],
                         "metrics": metrics_dict,
                         "feature_freq": res["feature_freq"]}, cache_path)
            saes[(site, exp)] = {"state": res["state"], "config": res["config"],
                                  "metrics": metrics_dict}
            print(f"[sae]   {site} x{exp}: EV={res['explained_var']:.4f}, "
                  f"L0={res['sparsity_l0']:.1f}, "
                  f"dead={res['dead_feature_rate']:.2%} "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return saes


# -------------------- Panels --------------------

def panel_a_decoder_cosine(saes_by_model: dict, sites: list[str],
                             expansion: int = 8) -> dict:
    """Panel A — for each (i, j, site) decoder-cosine histogram + summary stats."""
    results = {}
    for site in sites:
        results[site] = {}
        for i, mi in enumerate(MODELS):
            if (site, expansion) not in saes_by_model[mi]:
                continue
            for j, mj in enumerate(MODELS):
                if i >= j or (site, expansion) not in saes_by_model[mj]:
                    continue
                a = saes_by_model[mi][(site, expansion)]["state"]
                b = saes_by_model[mj][(site, expansion)]["state"]
                key = f"seed{i+1}_vs_seed{j+1}"
                results[site][key] = decoder_cosine_histogram(a, b, threshold=0.5)
    # Aggregate: mean fraction_stable per site across all pairs
    summary = {}
    for site, by_pair in results.items():
        if not by_pair:
            continue
        fracs = [v["fraction_stable"] for v in by_pair.values()]
        means = [v["mean_max_cos"] for v in by_pair.values()]
        summary[site] = {
            "n_pairs": len(by_pair),
            "mean_fraction_stable": float(np.mean(fracs)),
            "median_fraction_stable": float(np.median(fracs)),
            "mean_mean_max_cos": float(np.mean(means)),
        }
    return {"per_site_pair": results, "summary": summary}


def panel_b_cross_seed_encoder_ev(saes_by_model: dict, sites: list[str],
                                     eval_acts_by_model: dict,
                                     expansion: int = 8) -> dict:
    """Panel B — cross-seed encoder EV: apply SAE_i to acts_j."""
    results = {}
    for site in sites:
        results[site] = {}
        for i, mi in enumerate(MODELS):
            if (site, expansion) not in saes_by_model[mi]:
                continue
            sae_state = saes_by_model[mi][(site, expansion)]["state"]
            sae_cfg = saes_by_model[mi][(site, expansion)]["config"]
            # Apply to every seed's eval activations
            for j, mj in enumerate(MODELS):
                acts_j = eval_acts_by_model[mj][site].reshape(
                    -1, eval_acts_by_model[mj][site].shape[-1])
                m = sae_metrics(sae_state, sae_cfg, acts_j, device="cuda")
                key = f"sae{i+1}_on_acts{j+1}"
                results[site][key] = m
    return results


def panel_c_rotation_audit(saes_by_model: dict, sites: list[str],
                             eval_acts_by_model: dict,
                             expansion: int = 8) -> dict:
    """Panel C — one-batch Procrustes between seed_i and seed_j; post-rotation EV."""
    results = {}
    anchor_idx = 0  # seed1 is the anchor; all other seeds rotated to it
    for site in sites:
        results[site] = {"alignment": {}, "sae_post_rotation": {}}
        if anchor_idx >= len(MODELS):
            continue
        m_anchor = MODELS[anchor_idx]
        acts_anchor = eval_acts_by_model[m_anchor][site].reshape(
            -1, eval_acts_by_model[m_anchor][site].shape[-1])
        if (site, expansion) not in saes_by_model[m_anchor]:
            continue
        sae_anchor_state = saes_by_model[m_anchor][(site, expansion)]["state"]
        sae_anchor_cfg = saes_by_model[m_anchor][(site, expansion)]["config"]
        for j, mj in enumerate(MODELS):
            if j == anchor_idx:
                continue
            acts_j = eval_acts_by_model[mj][site].reshape(
                -1, eval_acts_by_model[mj][site].shape[-1])
            # Procrustes: find R that maps acts_j to acts_anchor
            R = best_orthogonal(acts_j, acts_anchor)
            metrics = procrustes_metrics(acts_j, acts_anchor, R)
            key = f"seed{j+1}_to_anchor"
            results[site]["alignment"][key] = metrics
            # Apply anchor SAE to (acts_j @ R) — centering matters
            mean_j_rotated = acts_j.mean(0) @ R
            mean_anchor = acts_anchor.mean(0)
            shift = mean_anchor - mean_j_rotated
            acts_j_rotated = acts_j @ R + shift
            m_raw = sae_metrics(sae_anchor_state, sae_anchor_cfg, acts_j,
                                  device="cuda")
            m_rot = sae_metrics(sae_anchor_state, sae_anchor_cfg, acts_j_rotated,
                                  device="cuda")
            results[site]["sae_post_rotation"][key] = {
                "raw": m_raw, "rotated": m_rot,
            }
        # Site-level summary
        if results[site]["alignment"]:
            raw_evs = [results[site]["sae_post_rotation"][k]["raw"]["explained_var"]
                       for k in results[site]["sae_post_rotation"]]
            rot_evs = [results[site]["sae_post_rotation"][k]["rotated"]["explained_var"]
                       for k in results[site]["sae_post_rotation"]]
            frobs = [results[site]["alignment"][k]["frob_R_minus_I"]
                     for k in results[site]["alignment"]]
            results[site]["summary"] = {
                "n_pairs": len(raw_evs),
                "mean_raw_EV": float(np.mean(raw_evs)),
                "mean_rot_EV": float(np.mean(rot_evs)),
                "mean_frob_R_minus_I": float(np.mean(frobs)),
                "median_frob_R_minus_I": float(np.median(frobs)),
                "predicted_random_orthogonal_frob": float(
                    (2 * eval_acts_by_model[m_anchor][site].shape[-1]) ** 0.5),
            }
    return results


# -------------------- Panel D — Steering --------------------

@torch.no_grad()
def build_sentiment_direction(model, tok, layer_idx: int, device: str = "cuda"):
    """diff-of-means direction at layer layer_idx between positive and negative
    sentiment sentences. Returns [d_model] vector on CPU."""
    pos = [
        "I absolutely loved this movie, it was wonderful and brilliant.",
        "An amazing experience, highly recommended to everyone, fantastic.",
        "What a delightful, joyful, fantastic, perfect, lovely creation.",
        "This was the best book I have ever read, truly excellent and inspiring.",
        "The service was outstanding, friendly, helpful, and absolutely perfect.",
    ] * 4
    neg = [
        "I hated this movie, it was terrible, awful, and boring.",
        "A horrible experience, do not recommend, dreadful and disappointing.",
        "What a miserable, painful, awful, terrible, bad creation.",
        "This was the worst book I have ever read, absolutely terrible and dull.",
        "The service was awful, rude, unhelpful, and completely disappointing.",
    ] * 4
    def encode(sents):
        outs = []
        for s in sents:
            ids = tok(s, return_tensors="pt").input_ids.to(device)
            o = model(ids, output_hidden_states=True, use_cache=False)
            outs.append(o.hidden_states[layer_idx + 1].float().mean(dim=(0, 1)).cpu())
        return torch.stack(outs, dim=0).mean(0)
    return (encode(pos) - encode(neg))


@torch.no_grad()
def build_name_direction(model, tok, layer_idx: int, device: str = "cuda"):
    """diff-of-means direction at the slot of John vs Mary in IOI-style sentences."""
    sentences_john = [
        "Today, John and Mary went to the store. Mary gave the milk to John.",
        "Yesterday, John and Mary saw a movie. The popcorn was given to John.",
        "After lunch, John and Mary talked. Mary gave the keys to John.",
    ] * 4
    sentences_mary = [
        "Today, John and Mary went to the store. John gave the milk to Mary.",
        "Yesterday, John and Mary saw a movie. The popcorn was given to Mary.",
        "After lunch, John and Mary talked. John gave the keys to Mary.",
    ] * 4
    def encode(sents):
        outs = []
        for s in sents:
            ids = tok(s, return_tensors="pt").input_ids.to(device)
            o = model(ids, output_hidden_states=True, use_cache=False)
            # Use the last token's residual as the IOI signal
            outs.append(o.hidden_states[layer_idx + 1][:, -1, :].float().squeeze(0).cpu())
        return torch.stack(outs, dim=0).mean(0)
    return encode(sentences_john) - encode(sentences_mary)


@torch.no_grad()
def build_magnitude_direction(model, tok, layer_idx: int, device: str = "cuda"):
    """Diff-of-means: large numbers vs small numbers in counting sentences."""
    big = ["The total was 1000 dollars.", "She counted 999 items.", "We had 500 guests.",
           "There were 750 books.", "The bill was 800 euros."] * 4
    small = ["The total was 2 dollars.", "She counted 3 items.", "We had 5 guests.",
             "There were 4 books.", "The bill was 6 euros."] * 4
    def encode(sents):
        outs = []
        for s in sents:
            ids = tok(s, return_tensors="pt").input_ids.to(device)
            o = model(ids, output_hidden_states=True, use_cache=False)
            outs.append(o.hidden_states[layer_idx + 1].float().mean(dim=(0, 1)).cpu())
        return torch.stack(outs, dim=0).mean(0)
    return encode(big) - encode(small)


@torch.no_grad()
def measure_steering_effect(model, tok, vector: torch.Tensor, layer_idx: int,
                              tokens: torch.Tensor, alpha: float,
                              device: str = "cuda",
                              chunk_seqs: int = 8) -> dict:
    """Apply the steering vector at layer_idx during forward and measure
    KL between (steered logits, clean logits) averaged over batch.

    vector: [d_model] tensor on CPU.
    layer_idx: which block's output gets the addition.
    chunk_seqs: process in this many sequences at a time to fit in VRAM (the
        full vocab×T×B logit tensor is ~6 GB at d_model=512, T=256, vocab=50k,
        which OOMs on a 12GB RTX 2060 when doubled to fp64 for KL precision).
    """
    import torch.nn.functional as F

    from .ig_pythia import _gpt_neox_blocks
    blocks = _gpt_neox_blocks(model)
    v = (alpha * vector).to(device)

    def make_hook():
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + v.to(output[0].dtype),) + output[1:]
            return output + v.to(output.dtype)
        return hook

    total_kl = 0.0
    total_n = 0
    N = tokens.shape[0]
    for i in range(0, N, chunk_seqs):
        sub = tokens[i : i + chunk_seqs].to(device)
        # Clean
        with torch.no_grad():
            out_clean = model(sub, use_cache=False)
            clean_logits = out_clean.logits.float()
        # Steered
        h = blocks[layer_idx].register_forward_hook(make_hook())
        try:
            with torch.no_grad():
                out_s = model(sub, use_cache=False)
                steer_logits = out_s.logits.float()
        finally:
            h.remove()
        # KL in fp32 (was fp64 — that's the OOM driver)
        logp_c = F.log_softmax(clean_logits, dim=-1)
        logp_s = F.log_softmax(steer_logits, dim=-1)
        p_c = logp_c.exp()
        kl_per_pos = (p_c * (logp_c - logp_s)).sum(dim=-1)  # [B, T]
        total_kl += float(kl_per_pos.sum().item())
        total_n += kl_per_pos.numel()
        del out_clean, out_s, clean_logits, steer_logits
        del logp_c, logp_s, p_c, kl_per_pos
        torch.cuda.empty_cache()
    kl_mean = total_kl / max(total_n, 1)
    return {"kl_clean_vs_steered": float(kl_mean), "alpha": alpha}


def panel_d_steering(eval_tokens: torch.Tensor, device: str = "cuda",
                       layer_idx: int = 3) -> dict:
    """Build three steering vectors on the anchor, apply to all other seeds.
    Measure transfer ratio at multiple alphas. Steering layer chosen mid-stack
    (layer 3 of 6 for Pythia-70m)."""
    print(f"[panel D] loading anchor {ANCHOR} ...", flush=True)
    anchor_model, anchor_tok = load_pythia(ANCHOR, dtype="bf16", device=device)
    # Build vectors on the anchor
    v_sent = build_sentiment_direction(anchor_model, anchor_tok, layer_idx, device)
    v_name = build_name_direction(anchor_model, anchor_tok, layer_idx, device)
    v_mag = build_magnitude_direction(anchor_model, anchor_tok, layer_idx, device)
    # Subsample eval tokens for KL measurement (256 sequences, faster)
    n_kl = min(64, eval_tokens.shape[0])
    kl_tokens = eval_tokens[:n_kl]
    # Measure within-anchor effect
    alphas = [0.5, 1.0, 2.0, 4.0]
    within = {}
    for name, v in [("sentiment", v_sent), ("name", v_name), ("magnitude", v_mag)]:
        within[name] = {str(a): measure_steering_effect(
            anchor_model, anchor_tok, v, layer_idx, kl_tokens, a, device=device)
            for a in alphas}
    del anchor_model
    torch.cuda.empty_cache(); gc.collect()
    # Now measure cross-seed effect — load each other seed, apply the same vector
    cross = {name: {str(a): [] for a in alphas}
             for name in ("sentiment", "name", "magnitude")}
    for mj in MODELS:
        if mj == ANCHOR:
            continue
        print(f"[panel D] target {mj} ...", flush=True)
        model_j, tok_j = load_pythia(mj, dtype="bf16", device=device)
        for name, v in [("sentiment", v_sent), ("name", v_name), ("magnitude", v_mag)]:
            for a in alphas:
                r = measure_steering_effect(model_j, tok_j, v, layer_idx,
                                               kl_tokens, a, device=device)
                cross[name][str(a)].append(r["kl_clean_vs_steered"])
        del model_j
        torch.cuda.empty_cache(); gc.collect()
    # Aggregate
    out = {"layer_idx": layer_idx, "alphas": alphas,
            "within_anchor": within, "cross": cross}
    out["transfer_ratio"] = {}
    for name in ("sentiment", "name", "magnitude"):
        out["transfer_ratio"][name] = {}
        for a in alphas:
            w = within[name][str(a)]["kl_clean_vs_steered"]
            c = float(np.mean(cross[name][str(a)])) if cross[name][str(a)] else 0.0
            out["transfer_ratio"][name][str(a)] = c / max(w, 1e-12)
    return out


# -------------------- Main orchestrator --------------------

def run(panels: tuple[str, ...] = ("A", "B", "C", "D"),
        device: str = "cuda",
        n_sae_steps: int = 4000,
        expansions: tuple[int, ...] = (8,)):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("[exp1] preparing corpus ...", flush=True)
    tok, tokens_sae, tokens_eval, eval_cfg = prepare_token_batches(device=device)
    print(f"[exp1] models: {len(MODELS)} = {MODELS}", flush=True)

    # Sites: all 7 residual-stream sites for Pythia-70m (6 layers)
    # We use the d_model from the first model load
    print("[exp1] loading anchor to determine n_layers ...", flush=True)
    anchor, _ = load_pythia(ANCHOR, dtype="bf16", device=device)
    n_layers = model_n_layers(anchor)
    d_model = model_d_model(anchor)
    sites = site_list(n_layers)
    del anchor
    torch.cuda.empty_cache(); gc.collect()
    print(f"[exp1] n_layers={n_layers}, d_model={d_model}, sites={sites}", flush=True)

    results = {"models": MODELS, "anchor": ANCHOR,
                "sites": sites, "n_layers": n_layers, "d_model": d_model,
                "expansions": list(expansions),
                "corpus_cfg_sae": CORPUS_CFG_SAE.__dict__,
                "corpus_cfg_eval": eval_cfg.__dict__,
                "wall_sec": {}}

    # --- Train SAEs for every (model, site, expansion) ---
    print("[exp1] training/loading SAEs ...", flush=True)
    saes_by_model = {}
    t0 = time.time()
    for mi in MODELS:
        saes_by_model[mi] = train_saes_for_model(
            mi, sites, tokens_sae, CORPUS_CFG_SAE,
            expansions=expansions, n_steps=n_sae_steps, device=device)
    results["wall_sec"]["sae_training"] = time.time() - t0
    print(f"[exp1]   SAE training/loading done in "
          f"{results['wall_sec']['sae_training']:.1f}s", flush=True)

    # --- Collect eval activations ---
    print("[exp1] collecting eval activations ...", flush=True)
    eval_acts_by_model = {}
    t0 = time.time()
    for mi in MODELS:
        eval_acts_by_model[mi] = get_or_extract_acts(
            mi, tokens_eval, eval_cfg, sites, device=device)
    results["wall_sec"]["eval_acts"] = time.time() - t0

    # --- Panel A ---
    if "A" in panels:
        print("[exp1] PANEL A — decoder-cosine histograms ...", flush=True)
        t0 = time.time()
        results["panel_A"] = panel_a_decoder_cosine(saes_by_model, sites,
                                                       expansion=expansions[0])
        results["wall_sec"]["panel_A"] = time.time() - t0

    # --- Panel B ---
    if "B" in panels:
        print("[exp1] PANEL B — cross-seed encoder EV ...", flush=True)
        t0 = time.time()
        results["panel_B"] = panel_b_cross_seed_encoder_ev(
            saes_by_model, sites, eval_acts_by_model, expansion=expansions[0])
        results["wall_sec"]["panel_B"] = time.time() - t0

    # --- Panel C ---
    if "C" in panels:
        print("[exp1] PANEL C — rotation audit ...", flush=True)
        t0 = time.time()
        results["panel_C"] = panel_c_rotation_audit(
            saes_by_model, sites, eval_acts_by_model, expansion=expansions[0])
        results["wall_sec"]["panel_C"] = time.time() - t0

    # --- Panel D ---
    if "D" in panels:
        print("[exp1] PANEL D — diff-of-means steering ...", flush=True)
        t0 = time.time()
        results["panel_D"] = panel_d_steering(tokens_eval, device=device,
                                                  layer_idx=n_layers // 2)
        results["wall_sec"]["panel_D"] = time.time() - t0

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w") as f:
        # Replace numpy/Tensor objects with serialisable forms
        json.dump(results, f, indent=2, default=str)
    print(f"[exp1] wrote {out_path}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--panels", default="ABCD",
                    help="Subset of A,B,C,D to run")
    p.add_argument("--n_sae_steps", type=int, default=4000)
    p.add_argument("--expansion", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    run(panels=tuple(args.panels), device=args.device,
        n_sae_steps=args.n_sae_steps,
        expansions=(args.expansion,))
