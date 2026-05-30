"""Analysis-only replication path.

Assumes `python -m replicate fetch-artifacts` has populated `artifacts/`.
Runs every analysis script that consumes cached artifacts, in dependency
order, and produces the four paper figures + a numerical-summary report.

Expected wall time on RTX 2060 12 GB: ~30 minutes.
Expected wall time on CPU: ~2 hours (everything still runs; activations are
cached so no PyTorch forward passes through Pythia are needed for most steps).

Outputs land at:
  artifacts/scale/pythia_rotation/{panel_c_fast,eigenvalue_spectrum,firing_pattern,ig_pythia}.json
  artifacts/cross_seed/*.json
  artifacts/cross_checkpoint/*.json
  artifacts/independent_init/*.json
  artifacts/bar_p_joint/*.json
  artifacts/figures/figure{1,2,3,4}.pdf
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ._paths import ARTIFACTS, ensure_layout

# Each step is (label, module path, args, est_minutes). Modules are invoked as
# `python -m polymorphism.<module>` from the repo root so their internal
# relative `experiments/...` paths resolve to `artifacts/` via the symlink
# established by ensure_layout().
STEPS = [
    ("Pythia decoder-cosine + naive-EV + rotation audit (panel C)",
     "polymorphism.experiments.scale.pythia_panel_c_fast", [], 5),
    ("Pythia eigenvalue spectrum + KS test against Haar SO(d)",
     "polymorphism.experiments.scale.eigenvalue_spectrum", [], 2),
    ("Pythia per-feature firing-pattern correlations",
     "polymorphism.experiments.scale.firing_pattern", [], 3),
    ("Pythia IG vs attribution-patching at convergence",
     "polymorphism.experiments.scale.ig_pythia", [], 2),
    ("Toy cross-seed SAE reconstruction (Cohort A)",
     "polymorphism.experiments.cross_seed.exp1a_reconstruction", [], 2),
    ("Toy cross-seed firing-pattern overlap",
     "polymorphism.experiments.cross_seed.exp1b_firing_overlap", [], 2),
    ("Toy cross-seed rotation audit",
     "polymorphism.experiments.cross_seed.exp1d_rotation_audit", [], 2),
    ("Toy steering-vector three-regime transfer",
     "polymorphism.experiments.cross_seed.exp2_steering", [], 2),
    ("Cross-seed aggregate metrics",
     "polymorphism.experiments.cross_seed.aggregate", [], 1),
    ("Independent-init (Cohort B) analysis",
     "polymorphism.experiments.independent_init.analyze_indep", [], 1),
    ("Cross-checkpoint rotation (Dyck-3 + Pythia)",
     "polymorphism.experiments.cross_checkpoint.cross_ckpt", [], 2),
    ("Cayley-refined joint Bar P",
     "polymorphism.experiments.bar_p_joint.joint_align", [], 3),
    ("Regenerate paper figures",
     "replicate.figures", [], 1),
]


def _check_artifacts() -> bool:
    """Return True if essential cached artifacts are present."""
    required = [
        ARTIFACTS / "scale" / "cache",
        ARTIFACTS / "seeds" / "0",
    ]
    missing = [p for p in required if not p.exists() or not any(p.iterdir() if p.is_dir() else [True])]
    if missing:
        print("[run-fast] missing required artifacts:", flush=True)
        for p in missing:
            print(f"  - {p}", flush=True)
        print("\nRun `python -m replicate fetch-artifacts` first.", flush=True)
        return False
    return True


def run_fast(skip_check: bool = False, only: list[str] | None = None) -> int:
    ensure_layout()
    if not skip_check and not _check_artifacts():
        return 2

    repo_root = Path(__file__).resolve().parent.parent
    total_est = sum(s[3] for s in STEPS)
    print(f"[run-fast] {len(STEPS)} steps, estimated {total_est} min total\n", flush=True)

    t0 = time.time()
    for i, (label, module, extra, est) in enumerate(STEPS, 1):
        if only and not any(o in module for o in only):
            print(f"[{i}/{len(STEPS)}] SKIP {label}", flush=True)
            continue
        print(f"\n[{i}/{len(STEPS)}] {label} (~{est} min)", flush=True)
        print(f"  $ python -m {module} {' '.join(extra)}", flush=True)
        step_t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", module, *extra],
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            print(f"\n[run-fast] step {i} failed (exit {result.returncode})", flush=True)
            return result.returncode
        print(f"  ok ({time.time() - step_t0:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n[run-fast] all steps completed in {elapsed / 60:.1f} min", flush=True)
    print(f"[run-fast] results under: {ARTIFACTS}", flush=True)
    print("[run-fast] verify with: python -m replicate verify", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-check", action="store_true",
                   help="skip the artifact-presence check before running")
    p.add_argument("--only", nargs="+", default=None,
                   help="run only steps whose module path contains any of these substrings")
    args = p.parse_args()
    return run_fast(skip_check=args.skip_check, only=args.only)


if __name__ == "__main__":
    sys.exit(main())
