"""Aggregate the cross-seed experiment results into one summary JSON
and write a concise markdown report.
"""

from __future__ import annotations

import json
import os

import numpy as np

CROSS_SEED_DIR = "experiments/cross_seed"


def _load(name: str):
    p = os.path.join(CROSS_SEED_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def main():
    a = _load("exp1a_reconstruction.json")
    b = _load("exp1b_firing_overlap.json")
    c = _load("exp1c_feature_patching.json")
    d = _load("exp1d_rotation_audit.json")
    e = _load("exp2_steering.json")

    summary = {
        "exp1a_reconstruction": {},
        "exp1b_firing_overlap": {},
        "exp1c_feature_patching": {},
        "exp1d_rotation_audit": {},
        "exp2_steering": {},
    }

    # 1a: per-site mean EV across seeds 1-4 vs seed 0
    if a:
        for site, exps in a.items():
            summary["exp1a_reconstruction"][site] = {}
            for exp_name, per_seed in exps.items():
                ev_self = per_seed.get("seed0", {}).get("explained_var", float("nan"))
                ev_cross = [per_seed.get(f"seed{s}", {}).get("explained_var")
                            for s in (1, 2, 3, 4)]
                ev_cross = [v for v in ev_cross if v is not None]
                ev_shuf = per_seed.get("control_shuffled_seed0", {}).get("explained_var", float("nan"))
                ev_2nd = per_seed.get("control_secondsample_seed0", {}).get("explained_var", float("nan"))
                ev_gau = per_seed.get("control_gaussian_seed0", {}).get("explained_var", float("nan"))
                summary["exp1a_reconstruction"][site][exp_name] = {
                    "EV_seed0_self": ev_self,
                    "EV_secondsample_seed0": ev_2nd,
                    "EV_cross_min": min(ev_cross) if ev_cross else None,
                    "EV_cross_max": max(ev_cross) if ev_cross else None,
                    "EV_cross_mean": float(np.mean(ev_cross)) if ev_cross else None,
                    "EV_shuffled": ev_shuf,
                    "EV_gaussian": ev_gau,
                }

    # 1b: per-site mean pearson cross-seed (raw and rotated)
    if b:
        for site, exps in b.items():
            summary["exp1b_firing_overlap"][site] = {}
            for exp_name, per_seed in exps.items():
                raw = [per_seed.get(f"seed{s}_raw", {}).get("mean_pearson_all")
                       for s in (1, 2, 3, 4)]
                raw = [v for v in raw if v is not None]
                rot = [per_seed.get(f"seed{s}_rotated", {}).get("mean_pearson_all")
                       for s in (1, 2, 3, 4)]
                rot = [v for v in rot if v is not None]
                self_p = per_seed.get("seed0_raw", {}).get("mean_pearson_all", float("nan"))
                summary["exp1b_firing_overlap"][site][exp_name] = {
                    "self_pearson": self_p,
                    "cross_raw_mean": float(np.mean(raw)) if raw else None,
                    "cross_rotated_mean": float(np.mean(rot)) if rot else None,
                    "cross_raw_min": min(raw) if raw else None,
                    "cross_rotated_min": min(rot) if rot else None,
                }

    # 1c: per-patch summary of transfer
    if c:
        for key, info in c.get("patches", {}).items():
            s0_kl = info.get("seed0_ablate", {}).get("kl_mean")
            cross_kls = [info.get(f"seed{s}_ablate", {}).get("kl_mean")
                         for s in (1, 2, 3, 4)]
            cross_kls = [v for v in cross_kls if v is not None]
            rand_kls = [info.get(f"seed{s}_random_dir", {}).get("kl_mean")
                        for s in (0, 1, 2, 3, 4)]
            rand_kls = [v for v in rand_kls if v is not None]
            target_head = ("valid" if info["target"] == "sticky_invalid"
                           else "depth" if info["target"].startswith("depth")
                           else "tok")
            s0_acc = info.get("seed0_ablate", {}).get(f"acc_{target_head}")
            cross_accs = [info.get(f"seed{s}_ablate", {}).get(f"acc_{target_head}")
                          for s in (1, 2, 3, 4)]
            cross_accs = [v for v in cross_accs if v is not None]
            summary["exp1c_feature_patching"][key] = {
                "target": info["target"], "feature": info["feature"],
                "corr": info["corr"],
                "seed0_KL": s0_kl,
                "cross_KL_mean": float(np.mean(cross_kls)) if cross_kls else None,
                "cross_KL_min":  min(cross_kls) if cross_kls else None,
                "cross_KL_max":  max(cross_kls) if cross_kls else None,
                "random_dir_KL_mean": float(np.mean(rand_kls)) if rand_kls else None,
                f"seed0_acc_{target_head}": s0_acc,
                f"cross_acc_{target_head}_mean":
                    float(np.mean(cross_accs)) if cross_accs else None,
                "transfer_ratio_KL": (
                    (float(np.mean(cross_kls)) / s0_kl)
                    if (cross_kls and s0_kl and s0_kl > 1e-6) else None),
            }

    # 1d: per-site rotation audit
    if d:
        for site, info in d.items():
            align = info.get("alignment", {})
            raw_evs = [v.get("raw_EV_vs_seed0") for v in align.values()
                       if v.get("raw_EV_vs_seed0") is not None]
            rot_evs = [v.get("rot_EV_vs_seed0") for v in align.values()
                       if v.get("rot_EV_vs_seed0") is not None]
            frob = [v.get("frob_R_minus_I") for v in align.values()
                    if v.get("frob_R_minus_I") is not None]
            sae = info.get("sae_post_rotation", {})
            sae_raw_x8 = [(sae.get(f"seed{s}", {}).get("x8_raw") or {}).get("explained_var")
                          for s in (1, 2, 3, 4)]
            sae_rot_x8 = [(sae.get(f"seed{s}", {}).get("x8_rotated") or {}).get("explained_var")
                          for s in (1, 2, 3, 4)]
            sae_raw_x8 = [v for v in sae_raw_x8 if v is not None]
            sae_rot_x8 = [v for v in sae_rot_x8 if v is not None]
            summary["exp1d_rotation_audit"][site] = {
                "raw_EV_mean":      float(np.mean(raw_evs)) if raw_evs else None,
                "rot_EV_mean":      float(np.mean(rot_evs)) if rot_evs else None,
                "frob_R_minus_I_mean": float(np.mean(frob)) if frob else None,
                "SAE_x8_EV_raw_mean":     float(np.mean(sae_raw_x8)) if sae_raw_x8 else None,
                "SAE_x8_EV_rotated_mean": float(np.mean(sae_rot_x8)) if sae_rot_x8 else None,
            }

    # 2: per-vector x alpha summary
    if e:
        for key, info in e.get("patches", {}).items():
            vec_name = info.get("vector")
            alpha = info.get("alpha")
            # KL
            s0_kl = info.get("seed0_v", {}).get("kl_mean")
            cross_kls = [info.get(f"seed{s}_v", {}).get("kl_mean")
                         for s in (1, 2, 3, 4)]
            cross_kls = [v for v in cross_kls if v is not None]
            rand_kls_s0 = info.get("seed0_random_dir", {}).get("kl_mean")
            rand_cross = [info.get(f"seed{s}_random_dir", {}).get("kl_mean")
                          for s in (1, 2, 3, 4)]
            rand_cross = [v for v in rand_cross if v is not None]
            # Conditional accuracy on positions the steering targets
            base_cond = info.get("seed0_v_cond_acc_base", {}).get("acc")
            s0_cond = info.get("seed0_v_cond_acc", {}).get("acc")
            cross_conds = [info.get(f"seed{s}_v_cond_acc", {}).get("acc")
                           for s in (1, 2, 3, 4)]
            cross_conds = [v for v in cross_conds if v is not None]
            summary["exp2_steering"][key] = {
                "vector": vec_name, "alpha": alpha,
                "seed0_KL": s0_kl,
                "cross_KL_mean": float(np.mean(cross_kls)) if cross_kls else None,
                "random_dir_KL_seed0": rand_kls_s0,
                "random_dir_KL_cross_mean": float(np.mean(rand_cross)) if rand_cross else None,
                "base_cond_acc": base_cond,
                "seed0_cond_acc_after": s0_cond,
                "cross_cond_acc_after_mean": float(np.mean(cross_conds)) if cross_conds else None,
                "seed0_drop": (base_cond - s0_cond) if (base_cond is not None and s0_cond is not None) else None,
                "cross_drop_mean": (
                    (base_cond - float(np.mean(cross_conds)))
                    if (base_cond is not None and cross_conds) else None),
                "transfer_ratio_drop": (
                    ((base_cond - float(np.mean(cross_conds))) / (base_cond - s0_cond))
                    if (base_cond is not None and s0_cond is not None
                        and cross_conds and (base_cond - s0_cond) > 1e-6) else None),
            }

    out_path = os.path.join(CROSS_SEED_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")
    return summary


if __name__ == "__main__":
    main()
