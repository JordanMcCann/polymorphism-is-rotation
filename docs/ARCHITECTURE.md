# Architecture notes

A short guide for the curious reader / contributor / reviewer.

## Two scales, one phenomenon

The paper makes a single claim — *cross-seed activations differ by an orthogonal rotation* — and tests it at two scales:

- **Toy (Dyck-3, 104k params).** Small enough that every weight has a documented role, so cross-seed comparisons can be done at the parameter level (Bars B, P, C, Pr).
- **Pythia-70m (71M params, public).** Activations only; the boundary between "interpretable model" and "rotation phenomenon" is here.

The repo mirrors this split:

- `src/polymorphism/model.py`, `task.py`, `train.py` — toy model and training
- `src/polymorphism/experiments/scale/` — Pythia-scale (activations + SAEs + rotation)
- `src/polymorphism/experiments/cross_seed/`, `cross_checkpoint/`, `independent_init/`, `bar_p_joint/` — cross-cutting experiments at toy scale

## The four bars and five lenses

Four falsifiable thresholds (Bars) and five complementary analyses (Lenses), one Python file per bar/lens. The bars are the methodological commitment of the paper; the lenses are the evidence underneath.

- **Bars** live in `src/polymorphism/verification/`: `bar_behavioral.py` (B), `bar_parametric.py` (P), `bar_causal.py` (C), `bar_predictive.py` (Pr). Each defines `run_bar_*(model, spec, ...) -> dict`.
- **Lenses** live in `src/polymorphism/analysis/`: `lens1_weights.py`, `lens2_saes.py`, `lens3_causal.py`, `lens4_polyhedral.py`, `lens5_constructive.py`. Each defines `run_lensN(model, ...) -> dict`.

`src/polymorphism/run_post_training.py` wires them together: load checkpoint → run all lenses → run all bars → write JSON.

## The orchestration layer

`replicate/` is a thin shell around the actual experiment scripts:

- `replicate/__main__.py` dispatches `python -m replicate <cmd>` to a sub-module.
- `replicate/run_fast.py` defines `STEPS = [(label, module_path, args, est_min), ...]` and runs them serially.
- `replicate/run_full.py` prepends training steps to that same list.
- `replicate/verify.py` defines `CLAIMS = [Claim(...), ...]` and walks them after the experiments are done.

Each experiment is a normal `python -m polymorphism.experiments.<name>` module — runnable independently — so the orchestration is non-magical. If something in `run-fast` fails, you can rerun just that script and diff the output.

## Why a directory symlink under the repo root

The legacy experiment code (which is exactly what produced the paper) writes to relative path `experiments/...`. To preserve that contract without rewriting every file, `_paths.py` creates a directory symlink `experiments/ -> artifacts/` on first run. This keeps the experiment scripts identical to what produced the published results while consolidating all outputs under a single `artifacts/` tree.

## Why uv

- Sub-second dependency resolution; full install in ~30 s once cached.
- `uv.lock` pins exact versions across platforms.
- Pip-compatible (`pip install -e ".[dev]"` works as a fallback).
- Native Windows support without WSL.

## Determinism boundaries

PyTorch is not bitwise deterministic across CUDA versions, cuDNN versions, or hardware. Three lines of defence:

1. Every random seed is fixed at the call site (`torch.manual_seed(seed)`, `np.random.seed(seed)`).
2. Where a computation is genuinely sample-based (Bar B at 1e7 samples), the CLT relative error is < 5% at our sample size.
3. The `replicate verify` tolerances (5-20% relative on most claims, exact on counts) absorb the residual hardware nondeterminism.

If you need bitwise determinism, set:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_DETERMINISTIC=1 \
    python -m replicate run-fast
```

(slows things down ~20%; rarely worthwhile)

## Outputs and the artifact contract

Every experiment writes:

1. A JSON to `artifacts/<subdir>/<exp_name>.json` containing every number the script computes (no "summary only"; the full table goes in)
2. PyTorch state-dicts to `artifacts/<subdir>/cache/` when training SAEs or models

Reading any JSON from a notebook is `json.load(open("artifacts/.../results.json"))` — no codec, no schema, no proprietary format.

## Tests

`tests/` covers the building blocks:

- `test_model.py` — forward pass, parameter count
- `test_task.py` — Dyck-3 sequence generation and labelling
- `test_rmsnorm_fold.py` — folding produces an equivalent model
- `test_symmetry_search.py` — alignment recovers the spec on synthetic data
- `test_scale_common.py` — Procrustes, cache key, decoder cosine
- `test_analysis_lenses.py` — each lens runs end-to-end on a tiny model
- `test_joint_align.py` — Cayley transform is orthogonal; joint loss is well-formed

`pytest tests/` runs all 43 in ~30 seconds. CI runs them on every PR.

The replication itself is also a kind of test — `replicate verify` is the integration-level test that the whole pipeline produces the paper's numbers.
