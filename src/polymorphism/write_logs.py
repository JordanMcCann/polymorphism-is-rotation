"""Auto-generate the convergence_log.md and verification_log.md content from
training and verification outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
else:
    pass


def load_seed_summary(seed: int) -> dict:
    """Load the final training summary for one seed."""
    log_path = f"logs/train_seed{seed}.json"
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        data = json.load(f)
    summary = None
    for r in data:
        if isinstance(r, dict) and "summary" in r:
            summary = r["summary"]
    return summary


def load_seed_final_eval(seed: int) -> dict:
    """Load the last evaluation result for one seed."""
    log_path = f"logs/train_seed{seed}.json"
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        data = json.load(f)
    last_eval = None
    for r in data:
        if isinstance(r, dict) and "train" in r and isinstance(r["train"], dict):
            last_eval = r
    return last_eval


def load_bar_results(seed: int) -> dict:
    """Load all 4 bar results for one seed."""
    out = {}
    bar_dir = f"experiments/seeds/{seed}/bar_outputs"
    for b in ("B", "P", "C", "Pr"):
        p = os.path.join(bar_dir, f"bar_{b}.json")
        if os.path.exists(p):
            with open(p) as f:
                out[b] = json.load(f)
    return out


def write_convergence_log(seeds: list[int] = None) -> str:
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    lines = [
        "# Convergence Log",
        "",
        "Cross-lens convergence record. Each component is described by every",
        "lens; this log tracks the iteration of disagreements and their",
        "resolutions.",
        "",
        "## Training convergence",
        "",
        "| Seed | Final step | acc_tok | acc_depth | acc_valid (train) | min held-out | first_pass | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for seed in seeds:
        summary = load_seed_summary(seed)
        last_eval = load_seed_final_eval(seed)
        if summary is None and last_eval is None:
            lines.append(f"| {seed} | _ | _ | _ | _ | _ | _ | not run |")
            continue
        if summary:
            final = summary.get("final", {})
            train_m = final.get("train", {})
            comp_m = final.get("compositional", {})
            long_m = final.get("long", {})
            n_ckpts = summary.get("n_ckpts", "?")
            first_pass = summary.get("first_pass_step", "_")
            step = summary.get("ckpt_paths", [""])[-1].split("ckpt_")[-1].split(".")[0]
        else:
            train_m = last_eval["train"]
            comp_m = last_eval["compositional"]
            long_m = last_eval["long"]
            n_ckpts = "?"
            first_pass = "(in progress)"
            step = last_eval["step"]
        min_held = min(
            comp_m.get("acc_tok", 1), comp_m.get("acc_depth", 1), comp_m.get("acc_valid", 1),
            long_m.get("acc_tok", 1), long_m.get("acc_depth", 1), long_m.get("acc_valid", 1),
        )
        lines.append(
            f"| {seed} | {step} | {train_m.get('acc_tok',0):.5f} | "
            f"{train_m.get('acc_depth',0):.5f} | {train_m.get('acc_valid',0):.5f} | "
            f"{min_held:.5f} | {first_pass} | _ |"
        )

    # Inter-lens table -- skeleton; filled by inspection
    lines.extend([
        "",
        "## Inter-lens agreement (primary seed)",
        "",
        "| Component | L1 (weights) | L2 (SAEs) | L3 (causal) | L4 (polyhedra) | L5 (compiled) | Status |",
        "|---|---|---|---|---|---|---|",
    ])

    lens_dir = f"experiments/seeds/{seeds[0]}/lens_outputs"
    lens1 = None
    if os.path.exists(os.path.join(lens_dir, "lens1.json")):
        with open(os.path.join(lens_dir, "lens1.json")) as f:
            lens1 = json.load(f)
    lens3 = None
    if os.path.exists(os.path.join(lens_dir, "lens3.json")):
        with open(os.path.join(lens_dir, "lens3.json")) as f:
            lens3 = json.load(f)
    lens4 = None
    if os.path.exists(os.path.join(lens_dir, "lens4.json")):
        with open(os.path.join(lens_dir, "lens4.json")) as f:
            lens4 = json.load(f)

    def cell(s):
        return s.replace("\n", " ").strip()[:80]

    components = []
    for L in range(2):
        for h in range(4):
            components.append(f"L{L}-H{h} attn")
        components.append(f"L{L} MLP")
    components.extend(["W_E", "W_pos", "W_U_tok", "W_U_depth", "W_U_valid"])

    if lens1:
        head_norms = {}
        for h_summary in lens1["heads"]:
            key = f"L{h_summary['layer']}-H{h_summary['head']} attn"
            head_norms[key] = (h_summary["qk"]["M_norm"], h_summary["ov"]["Z_norm"])
        mlp_alive = {}
        for m_summary in lens1["mlps"]:
            key = f"L{m_summary['layer']} MLP"
            in_norms = m_summary["in_norms"]
            alive = sum(1 for x in in_norms if x > 0.01)
            mlp_alive[key] = alive

    if lens3:
        ablation = lens3["mean_ablation"]["components"]
    else:
        ablation = {}

    for c in components:
        L1_col = "_"; L3_col = "_"; L4_col = "_"
        if lens1 and c in head_norms:
            qk, ov = head_norms[c]
            L1_col = f"|QK|={qk:.2f}, |OV|={ov:.2f}"
        if lens1 and c in mlp_alive:
            L1_col = f"alive={mlp_alive[c]}/256"
        # L3: ablation effects
        if c.startswith("L"):
            comp_l3_key = (
                c.replace("L", "attn_").replace("-H", "_h").replace(" attn", "")
                .replace("L", "mlp_").replace(" MLP", "")
            )
            # The substitution above is too clever; use a manual map
            if "H" in c:
                # "L0-H0 attn" -> "attn_0_h0"
                _, after = c.split("L", 1)
                L_n, hpart = after.split("-H")
                h_n = hpart.split(" ")[0]
                comp_l3_key = f"attn_{L_n}_h{h_n}"
            elif "MLP" in c:
                L_n = c.split("L")[1].split(" ")[0]
                comp_l3_key = f"mlp_{L_n}"
            else:
                comp_l3_key = None
            if comp_l3_key and comp_l3_key in ablation:
                L3_col = f"loss_delta={ablation[comp_l3_key]:.3f}"
        lines.append(f"| {c} | {L1_col} | _ | {L3_col} | {L4_col} | _ | _ |")

    lines.extend([
        "",
        "## Disagreements and resolutions",
        "",
        "(none recorded yet -- this section is filled out as the analysis runs.)",
        "",
    ])

    return "\n".join(lines)


def write_verification_log(seeds: list[int] = None) -> str:
    if seeds is None:
        seeds = [0, 1, 2, 3]
    lines = [
        "# Verification Log",
        "",
        "Numerical record of the four-bar verification per seed.",
        "",
        "## Bar B (Behavioral) -- KL divergence",
        "",
        "| Seed | mean KL | per-head: tok / depth / valid | per-dist: train / comp / long | n_samples | Passed (< 10⁻⁴) |",
        "|---|---|---|---|---|---|",
    ]
    for seed in seeds:
        bars = load_bar_results(seed)
        if "B" in bars:
            r = bars["B"]
            kl = r.get("mean_kl", "_")
            per_head = r.get("per_head_kl", {})
            per_dist = r.get("per_distribution", {})
            ph = " / ".join(f"{per_head.get(k, 0):.2e}" for k in ("tok", "depth", "valid"))
            pd = " / ".join(f"{(per_dist.get(k, {}).get('kl', 0)):.2e}"
                            for k in ("train", "compositional", "long"))
            passed = r.get("passed", "_")
            lines.append(f"| {seed} | {kl:.3e} | {ph} | {pd} | {r.get('n_samples', 0):,} | {passed} |")
        else:
            lines.append(f"| {seed} | _ | _ | _ | _ | _ |")

    lines.extend([
        "",
        "## Bar P (Parametric) -- per-entry MSE after symmetry alignment",
        "",
        "| Seed | max MSE | global MSE | Passed (< 10⁻³) |",
        "|---|---|---|---|",
    ])
    for seed in seeds:
        bars = load_bar_results(seed)
        if "P" in bars:
            r = bars["P"]
            lines.append(f"| {seed} | {r['max_per_tensor_mse']:.3e} | "
                          f"{r.get('global_mse', 0):.3e} | {r['passed']} |")
        else:
            lines.append(f"| {seed} | _ | _ | _ |")

    lines.extend([
        "",
        "## Bar C (Causal) -- Pearson r predicted vs measured",
        "",
        "| Seed | r (IG) | r (attribution) | r (spec-ablation) | best r | n_edges | Passed (> 0.99) |",
        "|---|---|---|---|---|---|---|",
    ])
    for seed in seeds:
        bars = load_bar_results(seed)
        if "C" in bars:
            r = bars["C"]
            lines.append(f"| {seed} | {r.get('pearson_r_ig', 0) or 0:.4f} | "
                          f"{r.get('pearson_r_attribution', 0) or 0:.4f} | "
                          f"{r.get('pearson_r_spec_ablation', 0) or 0:.4f} | "
                          f"{r['pearson_r']:.4f} | {r.get('n_edges', 0)} | {r['passed']} |")
        else:
            lines.append(f"| {seed} | _ | _ | _ | _ | _ | _ |")

    lines.extend([
        "",
        "## Bar Pr (Predictive) -- Pearson r predicted vs measured",
        "",
        "| Seed | r | n_components | Passed (> 0.99) |",
        "|---|---|---|---|",
    ])
    for seed in seeds:
        bars = load_bar_results(seed)
        if "Pr" in bars:
            r = bars["Pr"]
            lines.append(f"| {seed} | {r['pearson_r']:.4f} | "
                          f"{r.get('n_components', 0)} | {r['passed']} |")
        else:
            lines.append(f"| {seed} | _ | _ | _ |")

    lines.extend([
        "",
        "## Residual unexplained variance",
        "",
        "(filled out after Bar P; describes what the symmetry alignment cannot",
        "fully account for and the hypothesised cause.)",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    with open("logs/convergence_log.md", "w", encoding="utf-8") as f:
        f.write(write_convergence_log(seeds))
    with open("logs/verification_log.md", "w", encoding="utf-8") as f:
        f.write(write_verification_log(seeds))
    print("Wrote logs/convergence_log.md and logs/verification_log.md")


if __name__ == "__main__":
    main()
