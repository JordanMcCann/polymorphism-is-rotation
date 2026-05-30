"""Central path management for the replication pipeline.

All experiments write under `artifacts/` at the repo root. The directory is
created on first use and is .gitignored. Subpaths:

    artifacts/
      seeds/{0..4,100..104}/        Dyck-3 checkpoints, lens outputs, bar outputs
      scale/
        cache/                      Pythia activation tensors (large; ~9 GB)
        pythia_rotation/            Pythia panel results (small JSON)
      cross_seed/                   Toy-scale cross-seed analysis
      cross_checkpoint/             Within-run rotation analysis
      independent_init/             Cohort B analysis
      bar_p_joint/                  Cayley-refined Bar P
      figures/                      Regenerated paper figures

The legacy code (under `polymorphism.experiments.*`) hard-codes
`experiments/...` relative to the current working directory. We bridge by
running every replication command with cwd set to the repo root and a
symlink/passthrough that maps `experiments/` to `artifacts/`.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

# Legacy code expects `experiments/` relative to cwd. We use ARTIFACTS as the
# canonical location and create `experiments` as a symlink (or junction on
# Windows) to it inside the repo root if it doesn't already exist.
LEGACY_EXPERIMENTS = REPO_ROOT / "experiments"


def ensure_layout() -> None:
    """Create `artifacts/` and bridge legacy `experiments/` -> `artifacts/`."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for sub in [
        "seeds", "scale/cache", "scale/pythia_rotation",
        "cross_seed", "cross_checkpoint", "independent_init",
        "bar_p_joint", "figures",
    ]:
        (ARTIFACTS / sub).mkdir(parents=True, exist_ok=True)

    if not LEGACY_EXPERIMENTS.exists():
        try:
            os.symlink(ARTIFACTS, LEGACY_EXPERIMENTS, target_is_directory=True)
        except (OSError, NotImplementedError):
            # On Windows without dev-mode/admin, fall back to a directory
            # junction via mklink.
            import subprocess
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(LEGACY_EXPERIMENTS), str(ARTIFACTS)],
                check=False, capture_output=True,
            )


# Convenience constants for the orchestration scripts to consult.
SEEDS_DIR = ARTIFACTS / "seeds"
SCALE_CACHE = ARTIFACTS / "scale" / "cache"
SCALE_ROTATION = ARTIFACTS / "scale" / "pythia_rotation"
CROSS_SEED = ARTIFACTS / "cross_seed"
CROSS_CHECKPOINT = ARTIFACTS / "cross_checkpoint"
INDEP_INIT = ARTIFACTS / "independent_init"
BAR_P_JOINT = ARTIFACTS / "bar_p_joint"
FIGURES = ARTIFACTS / "figures"
SHARED_IO_INIT = ARTIFACTS / "shared_io_init.pt"
