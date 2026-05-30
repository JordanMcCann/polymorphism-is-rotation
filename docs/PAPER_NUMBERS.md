# Paper numbers → reproducer scripts

This table maps every numerical claim in the paper to the script that produces it and the JSON path where the value lives after running.

After `python -m replicate run-fast`, every value here should be re-derivable from the corresponding JSON. `python -m replicate verify` automatically checks a curated subset (the headline rotation/Haar claims).

## Section 7 — Primary-seed bars (Cohort A, seed 0)

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Bar B = 1.74e-4 | §7 table | `polymorphism.run_bars` | `artifacts/seeds/0/bar_outputs/bar_behavioral.json` → `mean_kl` |
| Bar B per-head (tok/depth/valid) | §7 table | same | same → `per_head_kl` |
| Bar C IG r > 0.9995 | §7 table | same | `artifacts/seeds/0/bar_outputs/bar_causal.json` → `pearson_r_ig` |
| Bar Pr IG r > 0.9995 | §7 table | same | `artifacts/seeds/0/bar_outputs/bar_predictive.json` → `pearson_r_ig` |
| Adversarial decoy Bar B = 0.226 | §7 ¶3 | `polymorphism.verification.decoy` | `artifacts/seeds/0/bar_outputs/decoy_verification.json` |

## Section 8 — Cross-seed universality (Cohort A)

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Cross-seed Bar B (5 pairs) | §8 table | `polymorphism.verification.universality` | `artifacts/cross_seed/universality.json` → `bar_b` |
| Cross-seed Bar P max MSE 0.49-0.58 (baseline) | §8 table | same | same → `bar_p_baseline_max_mse` |
| Cross-seed Bar P max MSE 0.264-0.298 (Cayley) | §8 table | `polymorphism.experiments.bar_p_joint.joint_align` | `artifacts/bar_p_joint/results.json` → `cayley_max_mse_per_pair` |
| Cross-seed Bar C r 0.52-0.69 | §8 table | universality | `artifacts/cross_seed/universality.json` → `bar_c_r` |
| Decoder cosine ≥ 98% at intermediate sites | §8.1 | `polymorphism.experiments.cross_seed.exp1d_rotation_audit` | `artifacts/cross_seed/exp1d_rotation_audit.json` → `decoder_cosine_fraction_above_0.5` |

## Section 8.3 — Pythia rotation audit (the headline)

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| 36 cross-seed pairs per site | §8.3 | `polymorphism.experiments.scale.pythia_panel_c_fast` | `artifacts/scale/pythia_rotation/panel_c_fast.json` → `n_pairs_per_site` |
| Mean max-cosine 0.91-0.93 | §8.3 | same | same → `per_site_mean_max_cosine` |
| Final-layer drop to 62% | §8.3 | same | same → `final_layer_fraction_above_0.5` |
| Mean cross-seed raw EV ∈ [-2.11, +0.75] | §8.3 | same | same → `per_site_mean_raw_ev` |
| Post-rotation EV 0.85-0.99 | §8.3 | same | same → `per_site_mean_post_rotation_ev` |
| ||R - I||_F mean = 31.99 | §8.3 | same | same → `mean_frob_R_minus_I` |
| ||R - I||_F p10-p90 = [31.94, 32.03] | §8.3 | same | same → `p10_p90_frob` |
| sqrt(2·512) prediction = 32.00 (to 0.1%) | §8.3 | same | same → `predicted_random_orthogonal_frob` |
| **KS stat = 0.0027** | §8.3 | `polymorphism.experiments.scale.eigenvalue_spectrum` | `artifacts/scale/pythia_rotation/eigenvalue_spectrum.json` → `summary.ks_stat_pooled_vs_haar` |
| **KS p = 1.000** | §8.3 | same | same → `summary.ks_pvalue_pooled_vs_haar` |
| 28,672 pooled eigenvalues | §8.3 | same | same → `summary.n_pooled_eigenvalues` |
| 5,120 Haar samples | §8.3 | same | same → `haar_baseline.n_pooled_haar_samples` |
| Pooled mean cos(theta) = 0.0006 | §8.3 | same | same → `summary.pooled_mean_cos_theta` |
| ||R - P_best||_F = 29.6 ± 0.03 | §8.3 | same | same → `summary.mean_perm_dist` ± `std_perm_dist` |
| 56 (pair, site) combinations | §8.3 | same | same → `summary.n_pair_site_combinations` |

## Section 8.4 — Independent-init Cohort B

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Cohort B Bar P max MSE 6.8-8.6 | §8.4 | `polymorphism.experiments.independent_init.analyze_indep` | `artifacts/independent_init/analyze_indep.json` → `bar_p_per_pair_max_mse` |
| Naive raw EV [-1.53, -0.58] | §8.4 | same | same → `naive_raw_ev_range` |
| Post-rotation EV [0.965, 0.987] | §8.4 | same | same → `post_rotation_ev_range` |
| ||R - I||_F [10.80, 11.25] vs sqrt(128)=11.31 | §8.4 | same | same → `frob_R_minus_I_range` |

## Section 8.5 — Firing-pattern overlap

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Toy raw r [0.03, 0.51] | §8.5 ¶2 | `polymorphism.experiments.cross_seed.exp1b_firing_overlap` | `artifacts/cross_seed/exp1b_firing_overlap.json` → `raw_correlation_range` |
| Toy post-rotation r [0.27, 0.74] | §8.5 ¶2 | same | same → `rotated_correlation_range` |
| Pythia mean raw r [0.009, 0.075] | §8.5 ¶3 | `polymorphism.experiments.scale.firing_pattern` | `artifacts/scale/pythia_rotation/firing_pattern.json` → `per_site_mean_raw_r` |
| Pythia mean post-rotation r [0.252, 0.437] | §8.5 ¶3 | same | same → `per_site_mean_rotated_r` |
| Fraction features cross-seed r > 0.5 | §8.5 ¶3 | same | same → `fraction_above_0.5` |

## Section 8.6 — Cross-checkpoint rotation

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Dyck-3 naive cross-checkpoint EV [0.90, 0.95] | §8.6 variant a | `polymorphism.experiments.cross_checkpoint.cross_ckpt` | `artifacts/cross_checkpoint/cross_ckpt.json` → `dyck.naive_ev_range` |
| Dyck-3 post-rotation EV [0.98, 0.999] | §8.6 variant a | same | same → `dyck.rotated_ev_range` |
| ||R-I||_F decreases 10.4 → 2.9 by depth | §8.6 variant a | same | same → `dyck.per_site_frob_R_minus_I` |
| Pythia step3000→step143000 self-train EV 0.907 | §8.6 variant b | same | same → `pythia.self_train_ev` |
| Pythia naive EV = -0.86 | §8.6 variant b | same | same → `pythia.naive_ev` |
| Pythia post-rotation EV = +0.73 | §8.6 variant b | same | same → `pythia.rotated_ev` |
| Pythia ||R-I||_F = 16.81 | §8.6 variant b | same | same → `pythia.frob_R_minus_I` |

## Section 8.7 — Three-regime steering (Cohort A)

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Depth-shift cross/within = 1.02 at α=2 | §8.7 ¶2 | `polymorphism.experiments.cross_seed.exp2_steering` | `artifacts/cross_seed/exp2_steering.json` → `depth_shift.cross_to_within_at_alpha_2` |
| Sticky-invalid ~4× dose offset | §8.7 ¶3 | same | same → `sticky_invalid.dose_factor` |
| Closer-signal cross/within = 8.43 at α=1.0 | §8.7 ¶4 | same | same → `closer_signal.cross_to_within_at_alpha_1` |

## Section 8.7 (continued) — Pythia steering collapse to "inverted"

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Transfer ratios 1.8-10.0 across vectors+α | §8.7 ¶5 | (run-full only) `pythia_rotation --panels D` | `artifacts/scale/pythia_rotation/results.json` → `panel_D` |

## Section 8.8 — Joint Bar P Cayley refinement

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Joint loss does NOT clear Bar P at any λ | §8.8 / §discussion | `polymorphism.experiments.bar_p_joint.joint_align` | `artifacts/bar_p_joint/results.json` → `lambda_sweep` |
| λ=0 weight-only Cayley: max MSE 0.58 → 0.28 | §8.8 | same | same → `lambda_0_max_mse_before`, `lambda_0_max_mse_after` |
| ~50% improvement | §8.8 | same | same → `relative_improvement` |

## Appendix B — All Bar B values

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Per-seed mean KL, tok, depth, valid, train, comp, long | Table in App B | `polymorphism.run_bars` for each seed | `artifacts/seeds/{0..4,100..104}/bar_outputs/bar_behavioral.json` |

## Appendix C — IG vs AP at Pythia

| Paper number | Where in paper | Producer | JSON path |
|---|---|---|---|
| Per-block measured loss-delta, AP prediction, IG prediction | Table in App C | `polymorphism.experiments.scale.ig_pythia` | `artifacts/scale/pythia_rotation/exp5_ig_pythia.json` → `per_layer` |
| r(AP) = 0.05, r(IG) = 0.98 | Table caption | same | same → `pearson_r_ap`, `pearson_r_ig_n32` |
| Baseline CE = 3.134 nats | Table caption | same | same → `baseline_ce_nats` |
| Wall clock 34.6 s | Table caption | same | same → `wall_time_sec` |

---

If you find a paper number missing from this table or a mismatch between paper and JSON, please open an issue. The replication suite is supposed to satisfy the principle "every number is derivable from one script invocation."
