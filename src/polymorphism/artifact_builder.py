"""Build the static-HTML interactive artifact from lens & bar outputs.

For a given seed, gather:
   - the trained checkpoint (weights)
   - the spec-compiled model (weights)
   - lens outputs (lens1.json, lens3.json, lens4.json, lens5.json, SAE summaries)
   - bar outputs (bar_B/P/C/Pr.json)
   - the algorithm.md / algorithm.rasp text

and produce the JSON files in ARTIFACT/data/ that the explorer.js loads.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from polymorphism.model import Config, make_model
    from polymorphism.rmsnorm_fold import fold_rmsnorm
else:
    from .model import Config, make_model
    from .rmsnorm_fold import fold_rmsnorm


def _np(t: torch.Tensor) -> list:
    return t.detach().cpu().float().numpy().tolist()


def export_weights(model, dst_path: str):
    """Export weight tensors as a single JSON blob the browser can load."""
    folded = fold_rmsnorm(model)
    out = {
        "config": {k: v for k, v in folded.cfg.__dict__.items()},
        "params": {},
    }
    for name, p in folded.named_parameters():
        out["params"][name] = {"shape": list(p.shape), "data": _np(p)}
    for name, b in folded.named_buffers():
        if name == "W_pos":
            out["params"][name] = {"shape": list(b.shape), "data": _np(b)}
    with open(dst_path, "w") as f:
        json.dump(out, f)


def load_seed_model(seed: int):
    ckpts = sorted(glob.glob(f"experiments/seeds/{seed}/checkpoints/ckpt_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints for seed {seed}")
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    cfg_dict = state.get("cfg", {})
    cfg = Config(**{k: v for k, v in cfg_dict.items() if k in Config.__dataclass_fields__})
    m = make_model(cfg); m.load_state_dict(state["model_state"])
    return m, ckpts[-1]


def gather_lens_summary(seed: int) -> dict:
    out = {}
    lens_dir = f"experiments/seeds/{seed}/lens_outputs"
    for name in ("lens1.json", "lens3.json", "lens4.json", "lens5.json",
                  "lens2_summary.json", "transcoder_summary.json"):
        p = os.path.join(lens_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                out[name.replace(".json", "")] = json.load(f)
    return out


def gather_bar_summary(seed: int) -> dict:
    out = {}
    bar_dir = f"experiments/seeds/{seed}/bar_outputs"
    for name in ("bar_B", "bar_P", "bar_C", "bar_Pr", "summary"):
        p = os.path.join(bar_dir, f"{name}.json")
        if os.path.exists(p):
            with open(p) as f:
                out[name] = json.load(f)
    return out


def gather_examples(model, n: int = 16, device: str = "cpu") -> dict:
    """Run the model on N example sequences and capture token-by-token outputs."""
    from .task import TaskConfig, sample_batch
    model.eval().to(device)
    rng = np.random.default_rng(2024)
    task_cfg = TaskConfig(n_ctx=model.cfg.n_ctx)
    batch = sample_batch(n, task_cfg, rng, length_range=(4, 40))
    batch_dev = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out, cache = model(batch_dev["tok"], return_internals=True)
    examples = []
    for b in range(n):
        ex = {
            "tokens": batch["tok"][b].tolist(),
            "depth": batch["depth"][b].tolist(),
            "valid": batch["valid"][b].tolist(),
            "mask": batch["mask"][b].tolist(),
            "pred_tok": out["tok"][b].argmax(-1).tolist(),
            "pred_depth": out["depth"][b].argmax(-1).tolist(),
            "pred_valid": out["valid"][b].argmax(-1).tolist(),
        }
        examples.append(ex)
    return {"examples": examples, "n_ctx": model.cfg.n_ctx}


def build(seed: int = 0, out_dir: str = "ARTIFACT/data") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    model, ckpt = load_seed_model(seed)
    export_weights(model, os.path.join(out_dir, "weights.json"))

    # Compiled spec
    from .analysis.lens5_rasp import compile_spec_to_model
    spec_model, basis = compile_spec_to_model(model.cfg)
    export_weights(spec_model, os.path.join(out_dir, "spec_weights.json"))

    # Gather universality results if available
    universality_dir = "experiments/universality"
    univ_results = []
    if os.path.exists(universality_dir):
        for fn in sorted(os.listdir(universality_dir)):
            if fn.startswith("universality_") and fn.endswith(".json"):
                with open(os.path.join(universality_dir, fn)) as f:
                    univ_results.append(json.load(f))

    annotations = {
        "seed": seed,
        "ckpt_path": ckpt,
        "basis": basis,
        "model_config": model.cfg.__dict__,
        "lens": gather_lens_summary(seed),
        "bars": gather_bar_summary(seed),
        "universality": univ_results,
    }
    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=2, default=str)

    examples = gather_examples(model, n=32)
    with open(os.path.join(out_dir, "examples.json"), "w") as f:
        json.dump(examples, f, indent=2, default=str)

    # Spec doc
    spec_md_path = "spec/algorithm.md"
    if os.path.exists(spec_md_path):
        with open(spec_md_path) as f:
            spec_text = f.read()
        with open(os.path.join(out_dir, "spec.json"), "w") as f:
            json.dump({"markdown": spec_text}, f)

    # bars.json: structured per-bar summary (for the artifact's Bars tab)
    bars_summary = {}
    for k in ("B", "P", "C", "Pr"):
        r = annotations["bars"].get(f"bar_{k}")
        if r:
            entry = {
                "tolerance": r.get("tolerance"),
                "passed": r.get("passed"),
            }
            if "mean_kl" in r:
                entry["mean_kl"] = r["mean_kl"]
            if "max_per_tensor_mse" in r:
                entry["max_per_tensor_mse"] = r["max_per_tensor_mse"]
            if "pearson_r" in r:
                entry["pearson_r"] = r["pearson_r"]
            bars_summary[k] = entry
    with open(os.path.join(out_dir, "bars.json"), "w") as f:
        json.dump(bars_summary, f, indent=2, default=str)

    return annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="ARTIFACT/data")
    args = parser.parse_args()
    build(args.seed, args.out)
    print(f"Artifact data written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
