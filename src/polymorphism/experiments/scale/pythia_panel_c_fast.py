"""EXP 1 fast track — Panel C (rotation audit) only.

Skips the SAE training (which is the slow part of the full EXP 1 run).
Just extracts activations from all 9 Pythia seeds, fits a single Procrustes
R per (seed, anchor) pair, and reports:

  - mean explained variance with and without rotation
  - ||R-I||_F per pair (compare to sqrt(2*d_model)=32 for d_model=512)
  - histogram of decoder-direction-equivalents using raw activations

This is the headline "does the rotation hypothesis hold at Pythia scale?"
result. It runs in ~5-10 min total on GPU even with the 9 model loads.

Writes: experiments/scale/pythia_rotation/panel_c_fast.json
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
    load_cached_acts,
    load_pythia,
    model_d_model,
    model_n_layers,
    procrustes_metrics,
    save_cached_acts,
    site_list,
    stream_text_chunks,
    tokenize_corpus,
)

MODELS = [f"EleutherAI/pythia-70m-seed{i}" for i in range(1, 10)]
ANCHOR_IDX = 0
CACHE_DIR = "experiments/scale/cache"
OUT_PATH = "experiments/scale/pythia_rotation/panel_c_fast.json"


def main(device: str = "cuda", n_sequences: int = 256, seq_len: int = 256):
    from transformers import AutoTokenizer
    print(f"[panel_c_fast] {len(MODELS)} models, n_seq={n_sequences}, "
          f"seq_len={seq_len}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODELS[ANCHOR_IDX])
    cfg = CorpusConfig(n_sequences=n_sequences, seq_len=seq_len, seed=2026)
    print("[panel_c_fast] tokenizing corpus ...", flush=True)
    texts = stream_text_chunks(cfg)
    tokens = tokenize_corpus(texts, tok, cfg)
    print(f"[panel_c_fast] tokens: {tuple(tokens.shape)}", flush=True)

    # Probe layer count from first model
    print(f"[panel_c_fast] probing {MODELS[ANCHOR_IDX]} ...", flush=True)
    m_anchor, _ = load_pythia(MODELS[ANCHOR_IDX], dtype="bf16", device=device)
    n_layers = model_n_layers(m_anchor)
    d_model = model_d_model(m_anchor)
    sites = site_list(n_layers)
    print(f"[panel_c_fast] d_model={d_model} n_layers={n_layers} "
          f"sites={sites}", flush=True)
    del m_anchor
    torch.cuda.empty_cache(); gc.collect()

    # Extract eval activations from all 9 models, cache to disk
    all_acts: dict[str, dict[str, torch.Tensor]] = {}
    t0 = time.time()
    for mid in MODELS:
        cached = load_cached_acts(CACHE_DIR, mid, None, cfg, sites)
        if cached is not None:
            print(f"[panel_c_fast]   cache hit: {mid}", flush=True)
            all_acts[mid] = cached
            continue
        print(f"[panel_c_fast]   extracting {mid} ...", flush=True)
        m, _ = load_pythia(mid, dtype="bf16", device=device)
        ta = time.time()
        acts = collect_residual_activations(m, tokens, batch_size=16, device=device)
        save_cached_acts(CACHE_DIR, mid, None, cfg, acts)
        all_acts[mid] = acts
        del m
        torch.cuda.empty_cache(); gc.collect()
        print(f"[panel_c_fast]     done in {time.time() - ta:.1f}s", flush=True)
    print(f"[panel_c_fast] all extraction done in {time.time() - t0:.1f}s",
          flush=True)

    # Panel C: rotation audit
    anchor = MODELS[ANCHOR_IDX]
    results = {"models": MODELS, "anchor": anchor, "n_layers": n_layers,
                "d_model": d_model, "corpus_cfg": cfg.__dict__,
                "per_site": {}}
    for site in sites:
        results["per_site"][site] = {"pair_metrics": {}}
        a_anchor = all_acts[anchor][site].reshape(-1, d_model).float()
        for j, mj in enumerate(MODELS):
            if j == ANCHOR_IDX:
                continue
            a_j = all_acts[mj][site].reshape(-1, d_model).float()
            R = best_orthogonal(a_j, a_anchor)
            mtr = procrustes_metrics(a_j, a_anchor, R)
            results["per_site"][site]["pair_metrics"][f"seed{j+1}"] = mtr
        # Site-level summary
        ms = results["per_site"][site]["pair_metrics"].values()
        ms_list = list(ms)
        if ms_list:
            results["per_site"][site]["summary"] = {
                "n_pairs": len(ms_list),
                "mean_raw_EV": float(np.mean([m["raw_EV"] for m in ms_list])),
                "mean_rot_EV": float(np.mean([m["rot_EV"] for m in ms_list])),
                "median_rot_EV": float(np.median([m["rot_EV"] for m in ms_list])),
                "mean_frob_R_minus_I": float(np.mean([m["frob_R_minus_I"] for m in ms_list])),
                "median_frob_R_minus_I": float(np.median([m["frob_R_minus_I"] for m in ms_list])),
                "min_frob_R_minus_I": float(min(m["frob_R_minus_I"] for m in ms_list)),
                "max_frob_R_minus_I": float(max(m["frob_R_minus_I"] for m in ms_list)),
                "predicted_random_orthogonal_frob": float((2 * d_model) ** 0.5),
            }
            s = results["per_site"][site]["summary"]
            print(f"  {site:<22}  raw_EV={s['mean_raw_EV']:+.4f}  "
                  f"rot_EV={s['mean_rot_EV']:.4f}  "
                  f"||R-I||={s['mean_frob_R_minus_I']:.2f}  "
                  f"(pred {s['predicted_random_orthogonal_frob']:.1f})",
                  flush=True)

    # Compute all-pairs ||R-I||_F histogram across all (i, j) pairs and sites
    all_frobs = []
    for site in sites:
        for k, m in results["per_site"][site]["pair_metrics"].items():
            all_frobs.append(m["frob_R_minus_I"])
    results["all_pair_frob_histogram"] = {
        "values": all_frobs,
        "mean": float(np.mean(all_frobs)),
        "median": float(np.median(all_frobs)),
        "p10": float(np.percentile(all_frobs, 10)),
        "p90": float(np.percentile(all_frobs, 90)),
        "predicted_random_orthogonal_frob": float((2 * d_model) ** 0.5),
    }
    print(f"\n[panel_c_fast] all-pairs ||R-I||_F: "
          f"mean={results['all_pair_frob_histogram']['mean']:.2f} "
          f"vs predicted {results['all_pair_frob_histogram']['predicted_random_orthogonal_frob']:.1f}",
          flush=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[panel_c_fast] wrote {OUT_PATH}", flush=True)
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n_sequences", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=256)
    args = p.parse_args()
    main(device=args.device, n_sequences=args.n_sequences, seq_len=args.seq_len)
