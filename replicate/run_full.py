"""Full replication path: train from scratch, then run all analysis.

Trains:
  - Cohort A (seeds 0-4): 104k-param Dyck-3 transformer with shared frozen I/O
  - Cohort B (seeds 100-104): same architecture, independently initialised

Then runs the full analysis pipeline equivalent to `run-fast`, including
Pythia activation collection (downloads weights from HF if not cached).

Expected wall time on RTX 2060 12 GB: ~14 hours
  - 5 Cohort A seeds:        ~5 h
  - 5 Cohort B seeds:        ~5 h
  - Pythia activation cache: ~30 min
  - SAE training (toy + Pythia residual sites): ~3 h
  - Analysis + figures:      ~30 min

This is the reproducible-from-zero path. Most reviewers want `run-fast`.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ._paths import SHARED_IO_INIT, ensure_layout
from .run_fast import STEPS as FAST_STEPS

# Each step is (label, module, args, est_minutes), invoked as `python -m <module>`
# from the repo root. train_all writes to experiments/seeds/<seed>/ (a symlink to
# artifacts/seeds/) and logs/train_seed<seed>.json.
TRAIN_STEPS = [
    ("Train Cohort A seed 0 (random init; source of the shared frozen I/O basis)",
     "polymorphism.train_all", ["--seeds", "0"], 60),
    ("Extract shared frozen I/O from seed 0",
     "polymorphism.train_all",
     ["--extract_shared_io", "--source_seed", "0", "--out", str(SHARED_IO_INIT)], 1),
    ("Train Cohort A seeds 1-4 (shared frozen I/O)",
     "polymorphism.train_all",
     ["--seeds", "1,2,3,4", "--shared_io_init_path", str(SHARED_IO_INIT),
      "--freeze_shared_io"], 240),
    ("Train Cohort B seeds 100-104 (independent init)",
     "polymorphism.train_all", ["--seeds", "100,101,102,103,104"], 300),
    ("Run lenses + bars on all Dyck-3 seeds",
     "polymorphism.run_post_training",
     ["--seeds", "0,1,2,3,4,100,101,102,103,104", "--primary_seed", "0"], 60),
    ("Cache Pythia-70m activations for all 9 seeds",
     "polymorphism.experiments.scale.pythia_panel_c_fast", [], 30),
    ("Train SAEs on Pythia seed-1 residual sites",
     "polymorphism.experiments.scale.pythia_rotation", ["--panels", "ABCD"], 180),
]


def run_full() -> int:
    ensure_layout()
    repo_root = Path(__file__).resolve().parent.parent
    all_steps = TRAIN_STEPS + FAST_STEPS[:-1]  # FAST_STEPS last entry is figures; we keep it at end
    total_est = sum(s[3] for s in all_steps)
    print(f"[run-full] {len(all_steps) + 1} steps, estimated {total_est} min total ({total_est / 60:.1f} h)\n",
          flush=True)

    t0 = time.time()
    for i, (label, module, extra, est) in enumerate(all_steps, 1):
        print(f"\n[{i}/{len(all_steps) + 1}] {label} (~{est} min)", flush=True)
        print(f"  $ python -m {module} {' '.join(extra)}", flush=True)
        step_t0 = time.time()
        result = subprocess.run(
            [sys.executable, "-m", module, *extra],
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            print(f"\n[run-full] step {i} failed (exit {result.returncode})", flush=True)
            return result.returncode
        print(f"  ok ({(time.time() - step_t0) / 60:.1f} min)", flush=True)

    # Final: figures
    print(f"\n[{len(all_steps) + 1}/{len(all_steps) + 1}] Regenerate paper figures",
          flush=True)
    subprocess.run([sys.executable, "-m", "replicate.figures"],
                   cwd=str(repo_root), check=False)

    elapsed = time.time() - t0
    print(f"\n[run-full] all steps completed in {elapsed / 60:.1f} min", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.parse_args()
    return run_full()


if __name__ == "__main__":
    sys.exit(main())
