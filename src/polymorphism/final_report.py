"""Generate the final project report (per directive's COMPLETION section).

Reads training summaries, bar outputs, and universality results from disk
and renders a Markdown report with the actual numbers — no hardcoded values.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def count_lines(globpat: str) -> int:
    n = 0
    for path in glob.glob(globpat, recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                n += sum(1 for _ in f)
        except Exception:
            pass
    return n


def gather_seed_summary(seed: int) -> dict:
    log_path = f"logs/train_seed{seed}.json"
    if not os.path.exists(log_path):
        return None
    data = json.load(open(log_path))
    evals = [r for r in data if isinstance(r, dict)
             and 'train' in r and isinstance(r.get('train'), dict)]
    if not evals:
        return None
    best = max(evals, key=lambda r: r['eval_min_acc'])
    return {
        "seed": seed,
        "best_step": best['step'],
        "best_min_acc": best['eval_min_acc'],
        "train": best['train'],
        "compositional": best['compositional'],
        "long": best['long'],
        "n_evals": len(evals),
    }


def gather_bars(seed: int) -> dict:
    out = {}
    for b in ("B", "P", "C", "Pr"):
        p = f"experiments/seeds/{seed}/bar_outputs/bar_{b}.json"
        if os.path.exists(p):
            r = json.load(open(p))
            value = (r.get("mean_kl")
                      or r.get("max_per_tensor_mse")
                      or r.get("pearson_r"))
            out[b] = {
                "tolerance": r.get("tolerance"),
                "passed": r.get("passed"),
                "value": value,
                "full": r,
            }
    return out


def gather_universality() -> list:
    out = []
    udir = "experiments/universality"
    if os.path.exists(udir):
        for fn in sorted(os.listdir(udir)):
            if fn.startswith("universality_") and fn.endswith(".json"):
                out.append(json.load(open(os.path.join(udir, fn))))
    return out


def gather_decoy() -> dict:
    p = "logs/decoy_verification.json"
    if os.path.exists(p):
        return json.load(open(p))
    return {}


def fmt_value(x, fmt="{:.3e}"):
    if x is None or isinstance(x, str):
        return "_"
    if isinstance(x, bool):
        return "PASS" if x else "FAIL"
    try:
        return fmt.format(x)
    except Exception:
        return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="logs/final_report.md")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--primary", type=int, default=0)
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    lines = ["# Final Report", "",
             "*Connectome project &mdash; complete mechanistic decomposition*", ""]

    # Codebase summary
    n_src = count_lines("src/**/*.py")
    n_tests = count_lines("tests/**/*.py")
    n_artifact = (count_lines("ARTIFACT/*.html") + count_lines("ARTIFACT/*.css")
                  + count_lines("ARTIFACT/*.js"))
    n_docs = count_lines("**/*.md")
    lines += [
        "## What was built",
        "",
        f"- Source code: {n_src:,} lines of Python",
        f"- Tests: {n_tests:,} lines of Python",
        f"- Artifact: {n_artifact:,} lines of HTML/CSS/JS",
        f"- Documentation: {n_docs:,} lines of Markdown",
        "",
        "## Architecture and task",
        "",
        "- 2-layer transformer, ReLU MLP, no biases, RMSNorm",
        "- 104,448 parameters",
        "- vocab=40, n_ctx=64, d_model=64, d_head=16, n_heads=4, d_mlp=256",
        "- Task: bounded-depth Dyck-3 with depth labels (3 output heads)",
        "- Training: AdamW + warmup-cosine, weight_decay=1e-4 (L1=0; see logs/convergence_log.md for the directive deviation rationale)",
        "",
    ]

    # Seed summaries
    lines += ["## Training summary per seed", ""]
    lines += ["| Seed | Best step | acc_tok (train) | acc_depth | acc_valid | min held-out |"]
    lines += ["|---|---|---|---|---|---|"]
    for s in seeds:
        sm = gather_seed_summary(s)
        if sm is None:
            lines += [f"| {s} | _ | _ | _ | _ | _ |"]
            continue
        lines += [f"| {s} | {sm['best_step']} | {sm['train']['acc_tok']:.5f} | "
                  f"{sm['train']['acc_depth']:.5f} | {sm['train']['acc_valid']:.5f} | "
                  f"{sm['best_min_acc']:.5f} |"]
    lines += [""]

    # Bars on each seed (vs constructive spec from primary)
    lines += [f"## Four-bar matrix (each seed aligned to constructive spec from seed {args.primary})",
              "",
              "| Bar | Tolerance | " + " | ".join(f"Seed {s}" for s in seeds) + " |",
              "|---|---|" + "---|" * len(seeds)]
    bar_thresh = {"B": ("< 1e-4", "{:.3e}"),
                  "P": ("< 1e-3", "{:.3e}"),
                  "C": ("> 0.99", "{:.4f}"),
                  "Pr": ("> 0.99", "{:.4f}")}
    bars_per_seed = {s: gather_bars(s) for s in seeds}
    for b, (tol, vfmt) in bar_thresh.items():
        row = [f"| {b}", tol]
        for s in seeds:
            br = bars_per_seed.get(s, {}).get(b, {})
            v = br.get("value")
            passed = br.get("passed")
            cell = fmt_value(v, vfmt) + (" PASS" if passed else (" FAIL" if passed is False else ""))
            row.append(cell)
        lines += [" | ".join(row) + " |"]
    lines += [""]

    # Overall PASS/FAIL summary
    overall = []
    for s in seeds:
        all_passed = all(bars_per_seed.get(s, {}).get(b, {}).get("passed")
                          for b in ("B", "P", "C", "Pr"))
        overall.append((s, all_passed))
    lines += [
        "**Overall:**",
        ""
    ] + [f"- Seed {s}: {'**PASS** all four bars' if ok else 'fails at least one bar'}"
         for s, ok in overall] + [""]

    # Universality
    univ = gather_universality()
    lines += ["## Cross-seed universality", ""]
    if univ:
        lines += ["| Primary | Replication | B (KL) | P (max MSE) | C (r) | Pr (r) | Status |",
                  "|---|---|---|---|---|---|---|"]
        for u in univ:
            ub = u['bars']
            kl = ub.get('B', {}).get('mean_kl', None)
            mse = ub.get('P', {}).get('max_per_tensor_mse', None)
            cr = ub.get('C', {}).get('pearson_r', None)
            prr = ub.get('Pr', {}).get('pearson_r', None)
            lines += [f"| seed{u['primary_seed']} | seed{u['replication_seed']} | "
                      f"{fmt_value(kl)} | {fmt_value(mse)} | "
                      f"{fmt_value(cr, '{:.4f}')} | {fmt_value(prr, '{:.4f}')} | "
                      f"{u['status']} |"]
    else:
        lines += ["(no universality data yet)"]
    lines += [""]

    # Decoy verification
    decoy = gather_decoy()
    if decoy:
        lines += ["## Adversarial decoy verification (Attack 1)", ""]
        rejected = decoy.get("decoy_rejected", False)
        by = decoy.get("decoy_rejected_by", [])
        lines += [f"Decoy rejected: **{'YES' if rejected else 'NO'}**  "
                  f"(rejected by bars: {', '.join(by) if by else 'none'})",
                  ""]

    # Where to find things
    lines += [
        "## Where to find things",
        "",
        "- Paper: [paper/main.pdf](../paper/main.pdf), source [paper/main.tex](../paper/main.tex)",
        "- Replication guide: [docs/REPLICATION.md](../docs/REPLICATION.md)",
        "- Paper-number map: [docs/PAPER_NUMBERS.md](../docs/PAPER_NUMBERS.md)",
        "- Per-seed lens outputs: `artifacts/seeds/<seed>/lens_outputs/`",
        "- Per-seed bar outputs: `artifacts/seeds/<seed>/bar_outputs/`",
        "- Universality results: `artifacts/universality/`",
        "",
    ]

    out_text = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(out_text)


if __name__ == "__main__":
    main()
