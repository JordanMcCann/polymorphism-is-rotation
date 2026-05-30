"""EXP 5 — Pythia replication of the §3.3 IG-vs-attribution-patching finding.

The Dyck-3 paper's §3.3 shows that attribution patching anti-correlates with
measured patch effects at convergence (Pearson r in [-0.63, +0.58] across
five seeds), while integrated gradients gives r > 0.9995 on every seed.

Here we replicate on Pythia-70m-seed1 (a converged 70M-param Pythia checkpoint
trained on The Pile). At a much larger scale on natural language, the same
phenomenon should appear: a converged model has small gradients, so first-
order attribution-patching extrapolation is unreliable; IG with continuous
integration along (mean -> clean) recovers the true patch effect by the
completeness axiom.

Output: experiments/scale/pythia_rotation/exp5_ig_pythia.json with per-layer
predicted vs measured loss-delta, Pearson r for both predictors.
"""

from __future__ import annotations

import json
import os
import time

import torch

from .common import (
    CorpusConfig,
    load_pythia,
    stream_text_chunks,
    tokenize_corpus,
)
from .ig_pythia import run_ig_vs_ap_comparison


def run(model_id: str = "EleutherAI/pythia-70m-seed1",
        n_sequences: int = 32, seq_len: int = 128,
        n_ig_steps: int = 32, device: str = "cuda",
        out_path: str = "experiments/scale/pythia_rotation/exp5_ig_pythia.json"
        ) -> dict:
    print(f"[exp5] loading {model_id} (bf16) ...", flush=True)
    model, tok = load_pythia(model_id, dtype="bf16", device=device)
    print(f"[exp5] preparing corpus ({n_sequences} seq) ...", flush=True)
    cfg = CorpusConfig(n_sequences=n_sequences, seq_len=seq_len, seed=2026)
    texts = stream_text_chunks(cfg)
    tokens = tokenize_corpus(texts, tok, cfg)
    print(f"[exp5] tokens shape: {tokens.shape}", flush=True)
    # Run on a subsample to keep it tractable: 8 sequences for IG (since IG
    # does n_steps=32 forward+backward passes per layer per sample)
    sub = tokens[:8].to(device)
    t0 = time.time()
    print(f"[exp5] computing IG vs AP comparison (n_ig_steps={n_ig_steps}) ...",
          flush=True)
    res = run_ig_vs_ap_comparison(model, sub, n_ig_steps=n_ig_steps, device=device)
    res["wall_sec"] = time.time() - t0
    res["model_id"] = model_id
    res["n_sequences_eval"] = int(sub.shape[0])
    res["seq_len_eval"] = int(sub.shape[1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"[exp5] r_AP={res['pearson_r_ap']:.4f}  "
          f"r_IG={res['pearson_r_ig']:.4f}  "
          f"baseline_loss={res['baseline_loss']:.4f}  "
          f"wrote {out_path}", flush=True)
    return res


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model", default="EleutherAI/pythia-70m-seed1")
    p.add_argument("--n_ig_steps", type=int, default=32)
    args = p.parse_args()
    run(model_id=args.model, n_ig_steps=args.n_ig_steps, device=args.device)
