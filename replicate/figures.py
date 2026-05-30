"""Regenerate the four paper figures from cached experiment outputs.

Reads the per-experiment JSONs under `artifacts/` (scale/pythia_rotation/*.json,
cross_seed/*.json, seeds/*/bar_outputs/bar_C.json) and writes the four
publication PDFs to `artifacts/figures/` (override with --out-dir):

  figure1_sae_failure_and_recovery.pdf
  figure2_rotation_is_random.pdf
  figure3_steering_regimes.pdf
  figure4_ig_vs_ap.pdf

These are the same figures committed under `paper/`. Each figure is rendered
independently; if an input JSON is missing the figure is skipped with a
warning rather than aborting the whole step.

    python -m replicate figures                 # all four into artifacts/figures
    python -m replicate figures --out-dir /tmp  # somewhere else
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._paths import ARTIFACTS, FIGURES, ensure_layout

# Publication style — colour-blind safe, modest weight
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# Wong colourblind palette (https://www.nature.com/articles/nmeth.1618)
COLORS = {
    "self":        "#000000",    # black
    "raw":         "#D55E00",    # vermilion (red-orange)
    "rotated":     "#009E73",    # bluish green
    "predicted":   "#0072B2",    # blue
    "haar":        "#56B4E9",    # sky blue
    "permutation": "#E69F00",    # orange
    "neutral":     "#999999",    # grey
}


def _load(data: Path, relpath: str) -> dict:
    with open(data / relpath) as f:
        return json.load(f)


# ----------------- Figure 1: SAE failure + rotation recovery -----------------

def figure1(data: Path, fig_dir: Path):
    """Two-panel: Dyck toy and Pythia-70m, with self / raw cross / post-rotation EV."""
    # Dyck (cohort A) from exp1a_reconstruction.json (raw) + exp1d_rotation_audit.json (rotated)
    dyck_recon = _load(data, "cross_seed/exp1a_reconstruction.json")
    dyck_rot = _load(data, "cross_seed/exp1d_rotation_audit.json")
    pythia = _load(data, "scale/pythia_rotation/results.json")

    # Build per-site arrays for Dyck
    dyck_sites = [s for s in dyck_recon.keys()]
    dyck_self = []
    dyck_raw = []
    dyck_rotated = []
    for site in dyck_sites:
        s = dyck_recon[site]["x8"]
        dyck_self.append(s["seed0"]["explained_var"])
        cross_evs = [s[f"seed{i}"]["explained_var"] for i in range(1, 5)]
        dyck_raw.append(np.mean(cross_evs))
        # Post-rotation from rotation_audit, "x8_rotated"
        rot_evs = [
            dyck_rot[site]["sae_post_rotation"][f"seed{i}"]["x8_rotated"]["explained_var"]
            for i in range(1, 5)
        ]
        dyck_rotated.append(np.mean(rot_evs))

    # Pythia from results.json
    pythia_sites = list(pythia["panel_C"].keys())
    pyt_self = []
    pyt_raw = []
    pyt_rotated = []
    for site in pythia_sites:
        # Self from panel_B sae{N}_on_acts{N}
        b = pythia["panel_B"][site]
        # self EV: average over seed{i}_on_acts{i} for i in 1..9
        self_evs = []
        for i in range(1, 10):
            key = f"sae{i}_on_acts{i}"
            if key in b:
                self_evs.append(b[key]["explained_var"])
        pyt_self.append(np.mean(self_evs))
        # Raw cross-seed: average over (i, j) i != j with i = anchor = 1
        raw_evs = []
        for j in range(2, 10):
            key = f"sae1_on_acts{j}"
            if key in b:
                raw_evs.append(b[key]["explained_var"])
        pyt_raw.append(np.mean(raw_evs))
        # Rotated from panel_C
        c = pythia["panel_C"][site]["sae_post_rotation"]
        rot_evs = [
            c[f"seed{j}_to_anchor"]["rotated"]["explained_var"]
            for j in range(2, 10) if f"seed{j}_to_anchor" in c
        ]
        pyt_rotated.append(np.mean(rot_evs))

    # Site labels (concise)
    dyck_labels = ["pre0", "mid0", "post0", "pre1", "mid1", "post1", "pre2"]
    pyt_labels = ["L0pre", "L0", "L1", "L2", "L3", "L4", "L5"]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    bar_w = 0.27

    for ax, sites, self_v, raw_v, rot_v, labels, title in [
        (axes[0], dyck_sites, dyck_self, dyck_raw, dyck_rotated, dyck_labels,
         "Dyck-3 toy (d_model = 64)"),
        (axes[1], pythia_sites, pyt_self, pyt_raw, pyt_rotated, pyt_labels,
         "Pythia-70m (d_model = 512, 9 independent seeds)"),
    ]:
        x = np.arange(len(sites))
        ax.bar(x - bar_w, self_v, bar_w, label="self (within-seed)",
                color=COLORS["self"], alpha=0.85)
        ax.bar(x, raw_v, bar_w, label="naive cross-seed",
                color=COLORS["raw"], alpha=0.85)
        ax.bar(x + bar_w, rot_v, bar_w, label="post-rotation",
                color=COLORS["rotated"], alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=8)
        ax.set_ylabel("Explained variance")
        ax.set_title(title)
        ax.set_ylim(min(min(raw_v), -2.2), 1.05)
        # subtle gridlines only on y
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("Cross-seed SAE reconstruction: catastrophic naive, restored by rotation",
                  fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = fig_dir / "figure1_sae_failure_and_recovery.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ----------------- Figure 2: Rotation IS random orthogonal -----------------

def figure2(data: Path, fig_dir: Path):
    """Two-panel: ||R-I||_F per pair + eigenvalue spectrum vs Haar."""
    # Dyck rotation_audit
    dyck_rot = _load(data, "cross_seed/exp1d_rotation_audit.json")
    pythia = _load(data, "scale/pythia_rotation/results.json")
    eigspec = _load(data, "scale/pythia_rotation/eigenvalue_spectrum.json")

    # Panel A: ||R - I||_F observed, predicted sqrt(2d)
    # Toy d=64 → predicted 11.31. Get all observed values
    dyck_R_norms = []
    for site, d in dyck_rot.items():
        if site == "resid_pre_0":
            continue  # trivial site, R is arbitrary on null space
        for seed_key, vals in d["alignment"].items():
            dyck_R_norms.append(vals["frob_R_minus_I"])
    # Pythia: same, from panel_C
    pythia_R_norms = []
    for site, d in pythia["panel_C"].items():
        for seed_key, vals in d["alignment"].items():
            pythia_R_norms.append(vals["frob_R_minus_I"])

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # Panel A
    ax = axes[0]
    bins_dyck = np.linspace(8.5, 12.0, 26)
    bins_pyt = np.linspace(31.8, 32.2, 26)
    # Two side-by-side mini-histograms with shared x scaling tricks — use twin axes
    ax.hist(dyck_R_norms, bins=bins_dyck, color=COLORS["self"],
             alpha=0.65, label="Dyck-3 (d=64)", density=False)
    ax.axvline(np.sqrt(2 * 64), color=COLORS["self"], linestyle="--",
                linewidth=1.0, label=r"$\sqrt{2 \cdot 64} = 11.31$")
    ax.set_xlabel(r"$\|R - I\|_F$  (Dyck-3)")
    ax.set_ylabel("count (pair × site)")
    ax.set_title(r"Observed $\|R-I\|_F$ vs random-orthogonal prediction")
    ax.legend(loc="upper left", fontsize=7.5)

    # Inset for Pythia
    inset = ax.inset_axes([0.55, 0.20, 0.40, 0.55])
    inset.hist(pythia_R_norms, bins=bins_pyt, color=COLORS["rotated"],
                alpha=0.85, density=False)
    inset.axvline(np.sqrt(2 * 512), color=COLORS["rotated"], linestyle="--",
                   linewidth=1.0)
    inset.set_title(r"Pythia (d=512), $\sqrt{2 \cdot 512} = 32.00$",
                     fontsize=7.5)
    inset.set_xlabel(r"$\|R-I\|_F$", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.set_yticks([])
    inset.grid(alpha=0.2)

    # Panel B: eigenvalue spectrum at Pythia vs Haar
    ax = axes[1]
    s = eigspec
    edges = np.array(s["haar_baseline"]["edges"])
    haar_density = np.array(s["haar_baseline"]["density_per_eigenvalue"])
    obs_density = np.array(s["summary"]["pooled_angle_density"])
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax.bar(centres, obs_density, width=width * 0.85,
            color=COLORS["rotated"], alpha=0.75,
            label=f"observed (n={s['summary']['n_pooled_eigenvalues']:,})")
    ax.plot(centres, haar_density, color=COLORS["self"], linewidth=1.8,
             label=r"Haar SO(512) prediction")
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_xlabel(r"eigenvalue angle  $|\theta|$")
    ax.set_ylabel("density (per eigenvalue)")
    ks_stat = s["summary"]["ks_stat_pooled_vs_haar"]
    ks_p = s["summary"]["ks_pvalue_pooled_vs_haar"]
    ax.set_title(f"Eigenvalue spectrum of cross-seed R\n"
                  f"KS vs Haar: stat={ks_stat:.4f}, p={ks_p:.3f}")
    ax.legend(loc="upper left", fontsize=7.5)

    fig.suptitle("Cross-seed rotation R is a uniform random orthogonal matrix",
                  fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = fig_dir / "figure2_rotation_is_random.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ----------------- Figure 3: Three-regime steering -----------------

def figure3(data: Path, fig_dir: Path):
    """Steering dose-response: 3 vectors x 2 settings (Dyck Cohort A, Pythia)."""
    dyck = _load(data, "cross_seed/exp2_steering.json")
    pythia = _load(data, "scale/pythia_rotation/results.json")

    alphas = [0.5, 1.0, 2.0, 4.0]
    vectors_dyck = ["sticky_invalid_d4", "depth_4_to_5", "closer_signal"]
    labels_dyck = ["sticky-invalid suppress\n(partial)",
                   "depth shift\n(clean)",
                   "closer signal\n(inverted)"]

    # Dyck: condition-accuracy drop within and cross
    dyck_within = {v: [] for v in vectors_dyck}
    dyck_cross = {v: [] for v in vectors_dyck}
    for v in vectors_dyck:
        for a in alphas:
            key = f"{v}_alpha{a}"
            patch = dyck["patches"][key]
            within = patch["seed0_v_cond_acc_base"]["acc"] - patch["seed0_v_cond_acc"]["acc"]
            cross = []
            for s in range(1, 5):
                base = patch[f"seed{s}_v_cond_acc_base"]["acc"]
                steered = patch[f"seed{s}_v_cond_acc"]["acc"]
                cross.append(base - steered)
            dyck_within[v].append(within)
            dyck_cross[v].append(np.mean(cross))

    # Pythia: KL within and cross
    vectors_pyt = ["sentiment", "name", "magnitude"]
    pyt_within = {v: [] for v in vectors_pyt}
    pyt_cross = {v: [] for v in vectors_pyt}
    for v in vectors_pyt:
        for a in alphas:
            wa = pythia["panel_D"]["within_anchor"][v][str(a)]["kl_clean_vs_steered"]
            cross_pairs = [pythia["panel_D"]["cross"][v][str(s)][str(a)]["kl_clean_vs_steered"]
                            for s in range(2, 10) if str(s) in pythia["panel_D"]["cross"][v]]
            pyt_within[v].append(wa)
            pyt_cross[v].append(np.mean(cross_pairs))

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # Left: Dyck — three vectors, three regimes
    ax = axes[0]
    colors_d = [COLORS["raw"], COLORS["rotated"], COLORS["predicted"]]
    for i, v in enumerate(vectors_dyck):
        c = colors_d[i]
        ax.plot(alphas, dyck_within[v], "-o", color=c, linewidth=1.5,
                 markersize=4, label=f"{labels_dyck[i].split(chr(10))[0]} within")
        ax.plot(alphas, dyck_cross[v], "--s", color=c, linewidth=1.5,
                 markersize=4, alpha=0.7)
    ax.set_xlabel(r"steering magnitude $\alpha$")
    ax.set_ylabel("conditional-accuracy drop")
    ax.set_xscale("log", base=2)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_title("Dyck-3 (shared I/O): three regimes\nsolid = within; dashed = cross-seed")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(-0.05, 1.1)

    # Right: Pythia — single regime
    ax = axes[1]
    colors_p = [COLORS["raw"], COLORS["rotated"], COLORS["predicted"]]
    for i, v in enumerate(vectors_pyt):
        c = colors_p[i]
        ax.plot(alphas, pyt_within[v], "-o", color=c, linewidth=1.5,
                 markersize=4, label=f"{v} within")
        ax.plot(alphas, pyt_cross[v], "--s", color=c, linewidth=1.5,
                 markersize=4, alpha=0.7, label=f"{v} cross")
    ax.set_xlabel(r"steering magnitude $\alpha$")
    ax.set_ylabel("KL(clean ‖ steered)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(alphas)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_yscale("log")
    ax.set_title("Pythia-70m (no shared I/O): all inverted\nsolid = within; dashed = cross-seed")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    fig.suptitle("Steering transfer: three regimes with shared boundary, one without",
                  fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = fig_dir / "figure3_steering_regimes.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


# ----------------- Figure 4: IG vs AP at convergence -----------------

def figure4(data: Path, fig_dir: Path):
    """Scatter of predicted vs measured for AP and IG, on Dyck and Pythia."""
    # Dyck: bar_C.json for each seed has predicted_ig, predicted_attribution, measured
    seed_data = []
    for s in range(5):
        d = _load(data, f"seeds/{s}/bar_outputs/bar_C.json")
        comps = sorted(d["measured"].keys())
        meas = np.array([d["measured"][c] for c in comps])
        ig = np.array([d["predicted_ig"][c] for c in comps])
        ap = np.array([d["predicted_attribution"][c] for c in comps])
        seed_data.append((meas, ig, ap, s))

    # Pythia: exp5_ig_pythia.json
    pythia_ig = _load(data, "scale/pythia_rotation/exp5_ig_pythia.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))

    # Left: Dyck — all 5 seeds, AP vs IG
    ax = axes[0]
    all_meas_ig = []
    all_pred_ig = []
    all_meas_ap = []
    all_pred_ap = []
    for meas, ig, ap, _ in seed_data:
        all_meas_ig.extend(meas.tolist())
        all_pred_ig.extend(ig.tolist())
        all_meas_ap.extend(meas.tolist())
        all_pred_ap.extend(ap.tolist())
    all_meas_ig = np.array(all_meas_ig)
    all_pred_ig = np.array(all_pred_ig)
    all_meas_ap = np.array(all_meas_ap)
    all_pred_ap = np.array(all_pred_ap)

    ax.scatter(all_meas_ap, all_pred_ap, color=COLORS["raw"], s=18, alpha=0.7,
                label="attribution patching", marker="x")
    ax.scatter(all_meas_ig, all_pred_ig, color=COLORS["rotated"], s=18, alpha=0.85,
                label="integrated gradients (n=32)")
    lim_low = min(all_meas_ig.min(), all_pred_ig.min(),
                   all_pred_ap.min()) - 1
    lim_high = max(all_meas_ig.max(), all_pred_ig.max(),
                    all_pred_ap.max()) + 1
    ax.plot([lim_low, lim_high], [lim_low, lim_high], "k--", linewidth=0.8,
             alpha=0.5, label="y = x")
    ax.set_xlabel("measured loss-delta (mean ablation, nats)")
    ax.set_ylabel("predicted loss-delta (nats)")
    ax.set_title("Dyck-3 (5 seeds × 10 components)\n"
                  "AP r ∈ [−0.63, +0.58], IG r > 0.9995")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)

    # Right: Pythia — 6 layers, AP vs IG
    ax = axes[1]
    meas_p = np.array(pythia_ig["measured"])
    ig_p = np.array(pythia_ig["integrated_gradients"])
    ap_p = np.array(pythia_ig["attribution_patch"])

    ax.scatter(meas_p, ap_p, color=COLORS["raw"], s=40, alpha=0.85,
                label=f"AP (r = {pythia_ig['pearson_r_ap']:.2f})", marker="x")
    ax.scatter(meas_p, ig_p, color=COLORS["rotated"], s=40, alpha=0.9,
                label=f"IG (r = {pythia_ig['pearson_r_ig']:.2f})")
    lim_low = -0.5
    lim_high = max(meas_p.max(), ig_p.max()) + 0.5
    ax.plot([lim_low, lim_high], [lim_low, lim_high], "k--", linewidth=0.8,
             alpha=0.5, label="y = x")
    ax.set_xlabel("measured loss-delta (mean ablation, nats)")
    ax.set_ylabel("predicted loss-delta (nats)")
    ax.set_title("Pythia-70m (6 blocks)\nAP r = 0.05, IG r = 0.98")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(lim_low, lim_high)
    ax.set_ylim(lim_low, lim_high)

    fig.suptitle("Integrated gradients restores patch-effect predictability "
                  "lost by attribution patching at convergence",
                  fontsize=10.5, y=1.02)
    fig.tight_layout()
    out = fig_dir / "figure4_ig_vs_ap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


FIGURE_FNS = {
    1: figure1,
    2: figure2,
    3: figure3,
    4: figure4,
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=FIGURES,
                   help="directory to write figure{1,2,3,4}.pdf into (default: artifacts/figures)")
    p.add_argument("--data-dir", type=Path, default=ARTIFACTS,
                   help="directory holding the cached experiment JSONs (default: artifacts/)")
    p.add_argument("--figure", type=int, choices=[1, 2, 3, 4], default=None,
                   help="render only this one figure (default: all four)")
    args = p.parse_args()

    ensure_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[figures] data={args.data_dir}  ->  out={args.out_dir}", flush=True)

    which = [args.figure] if args.figure else [1, 2, 3, 4]
    n_fail = 0
    for n in which:
        try:
            FIGURE_FNS[n](args.data_dir, args.out_dir)
        except (FileNotFoundError, KeyError) as e:
            n_fail += 1
            print(f"[figures] figure{n} skipped: {type(e).__name__}: {e}", flush=True)

    pdfs = sorted(args.out_dir.glob("figure*.pdf"))
    print(f"[figures] {len(pdfs)} figure(s) present in {args.out_dir}:", flush=True)
    for f in pdfs:
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
