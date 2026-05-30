# Detailed replication guide

This doc walks through every step in detail. For the quickstart, see the top-level [README](../README.md).

## 1. Environment

Requires Python 3.10, 3.11, or 3.12. We use [uv](https://docs.astral.sh/uv/) for dependency management because it is fast, reproducible (`uv.lock`), and pip-compatible.

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# OR
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

uv sync --extra dev
```

If you prefer plain pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## 2. Fetch artifacts

```bash
python -m replicate fetch-artifacts
```

Downloads ~11 GB from `https://huggingface.co/datasets/Jordanfmccann/polymorphism-is-rotation` into `artifacts/`. Idempotent and resumable — Ctrl-C and re-run safely.

To download only one subset:

```bash
python -m replicate fetch-artifacts --only pythia   # ~9 GB
python -m replicate fetch-artifacts --only dyck     # ~2 GB
```

The artifacts contain:

| Subdir | Contents | Size |
|---|---|---|
| `scale/cache/` | Pythia-70m residual activations across 9 seeds × 7 sites | ~8.5 GB |
| `scale/cache/saes/` | Trained SAEs (x8 expansion) for every (seed, site) | ~1 GB |
| `scale/pythia_rotation/` | Result JSONs for each Pythia experiment | ~1 MB |
| `seeds/{0..4}/` | Cohort A trained models + lens outputs + bar outputs | ~1 GB |
| `seeds/{100..104}/` | Cohort B (independent init) | ~1 GB |
| `cross_seed/`, `cross_checkpoint/`, etc. | Toy-scale experiment results | ~1 MB |

## 3. Run the analysis (fast path)

```bash
python -m replicate run-fast
```

Runs 13 analysis steps:

1. Pythia decoder-cosine + naive-EV + rotation audit (panel C)
2. Pythia eigenvalue spectrum + KS test against Haar SO(d)
3. Pythia per-feature firing-pattern correlations
4. Pythia IG vs attribution-patching at convergence
5. Toy cross-seed SAE reconstruction
6. Toy cross-seed firing-pattern overlap
7. Toy cross-seed rotation audit
8. Toy steering-vector three-regime transfer
9. Cross-seed aggregate metrics
10. Independent-init (Cohort B) analysis
11. Cross-checkpoint rotation
12. Cayley-refined joint Bar P
13. Regenerate paper figures

Each step writes a JSON to `artifacts/` and the figures get written to `artifacts/figures/`. Total time on RTX 2060: ~30 minutes.

### Running individual steps

To run just one or two steps:

```bash
python -m replicate run-fast --only eigenvalue ig_pythia
```

(matches any step whose module path contains those substrings)

Or invoke a single experiment script directly:

```bash
python -m polymorphism.experiments.scale.eigenvalue_spectrum
```

Each script accepts `--help` and writes to a documented path under `artifacts/scale/pythia_rotation/`.

## 4. Verify

```bash
python -m replicate verify
```

Compares the most important numerical claims from the paper against the contents of `artifacts/scale/pythia_rotation/eigenvalue_spectrum.json` and friends. Exit code 0 iff every claim passes within tolerance.

For the **complete** list of paper numbers and their reproducer scripts, see [`PAPER_NUMBERS.md`](PAPER_NUMBERS.md).

## 5. Full retrain (optional)

If you do not trust the cached artifacts and want to reproduce *everything* from public weights only:

```bash
python -m replicate run-full
```

This trains:
- Cohort A seeds 0-4 (shared frozen I/O): ~5 h
- Cohort B seeds 100-104 (independent init): ~5 h
- Pythia activation cache: ~30 min
- All SAEs: ~3 h
- Analysis + figures: ~30 min

Total ~14 h on an RTX 2060. The output should match the cached artifacts to within BLAS / CUDA non-determinism noise (typically <1% relative error on the headline numbers).

## Determinism caveats

PyTorch is not bitwise deterministic across:
- CUDA versions
- cuDNN versions
- Hardware (different GPUs)
- BLAS implementations (MKL vs OpenBLAS)

The replication suite uses `torch.manual_seed` and `numpy.random.seed` everywhere, but residual nondeterminism in convolutions, attention, and SAE training can shift downstream numbers by a few percent. The verify tolerances are set to absorb this.

If you want bitwise determinism, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `PYTORCH_CUDA_DETERMINISTIC=1` before running — but this slows things down by ~20% and only matters if you suspect a real bug.

## Troubleshooting

### `fetch-artifacts` is slow or fails partway through

The download uses HF Hub which supports resumption. Just re-run — already-present files are skipped.

If you are behind a corporate proxy: set `HTTPS_PROXY` and `HF_HUB_DOWNLOAD_TIMEOUT=60` (or higher).

### `run-fast` fails on the eigenvalue spectrum step

Check that `artifacts/scale/cache/` is populated. If the `.npy` files are missing, re-run `fetch-artifacts --only pythia`.

If they are present but the script complains: confirm scipy is installed (`uv run python -c "import scipy"`).

### `verify` reports MISSING

This means the run-fast step that should have produced that JSON didn't complete (or didn't write the expected key). Look at the corresponding script:

```bash
python -m replicate verify        # see which claim is missing
ls -la artifacts/scale/pythia_rotation/
python -m polymorphism.experiments.scale.eigenvalue_spectrum   # rerun the step
python -m replicate verify
```

### CUDA out of memory

The Pythia analysis batch size is set conservatively (256 sequences × 256 tokens). If you still OOM, reduce:

```bash
python -m polymorphism.experiments.scale.pythia_panel_c_fast --n_sequences 64 --seq_len 128
```

Results will be noisier (smaller sample) but qualitatively identical.

### "ModuleNotFoundError: No module named 'polymorphism'"

You forgot `uv sync` (or didn't activate the venv). Re-run:

```bash
uv sync --extra dev
uv run python -m replicate help
```

## Where to file issues

https://github.com/JordanMcCann/polymorphism-is-rotation/issues — include OS, Python version, GPU, and the failing step.
