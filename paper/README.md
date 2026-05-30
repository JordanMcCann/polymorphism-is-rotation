# Paper source

LaTeX source and built PDF of the paper this repository replicates.

## Contents

| File | Description |
|---|---|
| `main.tex` | Paper source (single-file, no `\input`) |
| `refs.bib` | BibTeX bibliography (40 entries) |
| `main.bbl` | Pre-compiled bibliography (so arxiv doesn't need to rerun bibtex) |
| `main.pdf` | Built version of the paper (26 pages, ~520 KB) |
| `figure1_sae_failure_and_recovery.pdf` | Figure 1 |
| `figure2_rotation_is_random.pdf` | Figure 2 (KS test result, p ~= 1.000) |
| `figure3_steering_regimes.pdf` | Figure 3 (three-regime steering transfer) |
| `figure4_ig_vs_ap.pdf` | Figure 4 (IG vs attribution patching) |

## Rebuilding from source

```bash
cd paper/
pdflatex -interaction=nonstopmode main
bibtex main
pdflatex -interaction=nonstopmode main
pdflatex -interaction=nonstopmode main
```

Tested with TeX Live 2024 and MiKTeX 25.12. No special packages beyond a standard full TeX distribution (`amsmath`, `amssymb`, `graphicx`, `booktabs`, `microtype`, `hyperref`, `natbib`).

## Regenerating figures from data

The four figures are produced by `python -m replicate figures` from the cached JSON outputs:

```bash
# Run from repo root, not from paper/
python -m replicate figures                  # writes into artifacts/figures/
python -m replicate figures --out-dir paper  # OR render straight into paper/
```

The figure-rendering script reads from `artifacts/scale/pythia_rotation/*.json` and `artifacts/cross_seed/*.json`. Replication users typically don't need to regenerate figures since the PDFs above are committed; this is for the case where you've made code changes that should refresh them.
