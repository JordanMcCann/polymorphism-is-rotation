# polymorphism-is-rotation

Replication code for [*Polymorphism Is Rotation: Operational Mechanistic Interpretability from a Two-Layer Transformer to Pythia-70m*](paper/main.pdf) (McCann, 2026).

> Independently trained transformers compute the same function in residual-stream bases that differ by an essentially uniform random rotation on SO(d). One matrix multiplication per model pair removes it.

## Replicate in four commands

```bash
git clone https://github.com/JordanMcCann/polymorphism-is-rotation.git
cd polymorphism-is-rotation
uv sync                                       # install pinned deps (~2 min)
python -m replicate fetch-artifacts           # download cached Pythia + Dyck-3 outputs (~11 GB, ~10 min)
python -m replicate run-fast                  # recompute all analysis on the cached artifacts (~30 min GPU / ~2 h CPU)
python -m replicate verify                    # check every recomputed number against the paper's claims (~1 s)
```

If `verify` prints **6/6 claims verified**, you have independently recomputed the paper's
headline numerical claims from the cached activations and trained models. For a true
from-scratch run that retrains every model, use `run-full` (see below).

## Three replication paths

| Path | Time | What it does |
|------|------|---|
| `replicate verify` | 1 s | Compare existing `artifacts/` JSONs against paper claims. Useful right after `fetch-artifacts`. |
| `replicate run-fast` | ~30 min | Re-run every analysis script on **cached** Pythia activations + trained SAEs. Reproduces every paper figure and table from the artifacts. |
| `replicate run-full` | ~14 h | Train the 10 Dyck-3 seeds from scratch, cache Pythia activations from public weights, train all SAEs, run analysis. Zero external state needed. |

**Pick `run-fast`** — it is the verified path: a full `run-fast` → `verify` is exercised end-to-end and reproduces 6/6 (≈1 h on CPU, faster on GPU). **`run-full`** runs the same analysis after retraining every model from scratch; its individual stages are unit-tested, but the full ~14 h pipeline is not part of routine release testing, so treat it as the heavier, less-exercised tier. A divergence between `run-full` and the cached-artifact numbers is worth reporting.

> **Lighter option — confirm the numbers without the 11 GB fetch.** The [`v1.0.0` release](https://github.com/JordanMcCann/polymorphism-is-rotation/releases/tag/v1.0.0) attaches `polymorphism-replication-lite.tar.gz` (~13 MB): the final trained Dyck-3 checkpoints plus every result JSON (no activation cache). Extract it into `artifacts/` and run `verify` to check all six headline claims and inspect the trained models:
>
> ```bash
> mkdir -p artifacts && tar -xzf polymorphism-replication-lite.tar.gz -C artifacts
> python -m replicate verify        # -> 6/6 claims verified
> ```
>
> A full `run-fast` recompute still needs `python -m replicate fetch-artifacts` (it reads the cached Pythia activations, which aren't in the lite bundle).

## What gets reproduced

The headline claims of the paper, verified by `replicate verify`:

| Claim | Paper | Tolerance |
|---|---|---|
| KS statistic, R eigenvalues vs Haar SO(512) | 0.0027 | ±20% |
| KS p-value (pooled and per-pair) | 1.000 | ±1% |
| Pooled mean cos(theta) vs Haar prediction | 0.0006 | wide |
| `||R - best_perm||_F` mean across all 56 pairs | 29.6 | ±2% |
| (Pair, site) combinations | 56 | exact |
| Pooled eigenvalue count | 28,672 | exact |

Plus every figure (`artifacts/figures/figure{1,2,3,4}.pdf`), every per-experiment JSON under `artifacts/`, and the table of cross-seed Bar B/P/C/Pr values.

See [`docs/PAPER_NUMBERS.md`](docs/PAPER_NUMBERS.md) for the full map from each paper number to the script that produces it.

## Hardware

- **`run-fast`**: any machine with 16 GB RAM and 12 GB free disk. CUDA optional (CPU OK; ~2 h instead of 30 min).
- **`run-full`**: CUDA GPU with >= 8 GB VRAM strongly recommended. The paper's training + caching runs used an RTX 2060 12 GB and an RTX 3090; wall-time is roughly linear in compute.

## Layout

```
polymorphism-is-rotation/
├── README.md                 -- this file
├── pyproject.toml            -- uv-managed Python project (Python >=3.10)
├── uv.lock                   -- pinned dependency versions
├── replicate/                -- orchestration: fetch / run-fast / run-full / figures / verify
├── src/polymorphism/         -- the importable package
│   ├── model.py              -- 2-layer transformer (104k params)
│   ├── task.py               -- Dyck-3 generator + depth/validity labels
│   ├── train.py              -- AdamW + warmup-cosine, bf16
│   ├── rmsnorm_fold.py       -- analytical RMSNorm gain folding
│   ├── symmetry_search.py    -- alignment under the model's symmetry group
│   ├── analysis/             -- the five lenses (weights, SAEs, causal, polyhedral, constructive)
│   ├── verification/         -- the four bars (B, P, C, Pr) + adversarial decoy
│   └── experiments/
│       ├── cross_seed/       -- Cohort A cross-seed analysis (SAE transfer + steering)
│       ├── cross_checkpoint/ -- within-run rotation drift
│       ├── independent_init/ -- Cohort B independent-init replication
│       ├── bar_p_joint/      -- Cayley-refined Bar P
│       └── scale/            -- Pythia-70m (panel C, KS test, firing patterns, IG)
├── tests/                    -- pytest suite (49 tests; runs in ~30 s)
├── paper/                    -- the arxiv submission (main.tex, refs.bib, figures, built PDF)
├── docs/
│   ├── REPLICATION.md        -- detailed replication guide and troubleshooting
│   ├── PAPER_NUMBERS.md      -- table mapping every paper number to its reproducer script
│   └── ARCHITECTURE.md       -- explainer for repo layout and design choices
├── scripts/                  -- one-time maintainer scripts (HF Hub upload, etc.)
└── artifacts/                -- created on first run; populated by fetch-artifacts
```

## How the code is organised

- **`src/polymorphism/`** is a normal importable Python package. Use it from a notebook with `from polymorphism import model, task`, etc.
- **`replicate/`** is a thin orchestration layer that runs the experiments in dependency order. It writes everything under `artifacts/` and bridges to legacy `experiments/` paths via a directory symlink.
- All experiments write their outputs as JSON (small enough to commit / diff in PRs) plus PyTorch state-dicts for SAEs and trained models.

## The paper, alongside the code

The arxiv source is in [`paper/`](paper/) and the built PDF at [`paper/main.pdf`](paper/main.pdf). You can rebuild with `pdflatex && bibtex && pdflatex && pdflatex` if you have a TeX distribution.

## Citing

```bibtex
@article{mccann2026polymorphism,
  title   = {Polymorphism Is Rotation: Operational Mechanistic Interpretability
             from a Two-Layer Transformer to Pythia-70m},
  author  = {McCann, Jordan F.},
  journal = {arXiv preprint arXiv:2605.24577},
  year    = {2026}
}
```

## License

MIT. See [`LICENSE`](LICENSE).

## Issues / replication failures

Open an issue at https://github.com/JordanMcCann/polymorphism-is-rotation/issues. Include:
- Your OS, Python version, GPU (or "CPU only")
- The `python -m replicate verify` output
- The first failing step (run `--only <name>` to isolate it under `run-fast`)

The replication suite is designed so that any divergence from paper numbers is either a bug in the suite or genuinely new information about the rotation phenomenon. We'd like to hear about both.
