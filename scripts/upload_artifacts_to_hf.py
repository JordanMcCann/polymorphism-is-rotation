"""Upload cached artifacts to the Hugging Face Hub dataset repository.

This script is run ONCE by the project maintainer, not by replication users.
Users instead call `python -m replicate fetch-artifacts` which downloads from
the published dataset.

Requires the HF write token to be set via the HF_TOKEN environment variable.
The token is never written to disk by this script.

Usage:
    HF_TOKEN=hf_xxx python scripts/upload_artifacts_to_hf.py \\
        --source artifacts \\
        --repo-id Jordanfmccann/polymorphism-is-rotation
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DESCRIPTION = """\
Cached experiment artifacts for the paper *Polymorphism Is Rotation*
(McCann, 2026).

Contents:
- `scale/cache/` -- Pythia-70m residual-stream activations for 9 seeds
  across 7 residual sites (~8.5 GB), plus trained SAE checkpoints
  (`scale/cache/saes/`, ~1 GB).
- `scale/pythia_rotation/` -- per-experiment result JSONs (KS test,
  eigenvalue spectrum, firing patterns, IG vs AP).
- `seeds/{0..4,100..104}/` -- Dyck-3 trained model weights, lens outputs,
  bar outputs for Cohort A (shared frozen I/O) and Cohort B (independent
  init).
- `shared_io_init.pt` -- the seed-0 trained input/output weights frozen
  into Cohort A seeds 1-4.

Download via `python -m replicate fetch-artifacts` from the replication
repository at https://github.com/JordanMcCann/polymorphism-is-rotation.
"""


def upload(source: Path, repo_id: str, token: str, dry_run: bool = False) -> None:
    api = HfApi(token=token)

    print(f"[upload] ensuring repo {repo_id} exists ...", flush=True)
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
        token=token,
    )

    print(f"[upload] source: {source}", flush=True)
    if not source.exists():
        print(f"[upload] source does not exist; abort", flush=True)
        sys.exit(2)

    # Count what we're about to push.
    n_files = sum(1 for _ in source.rglob("*") if _.is_file())
    total_size = sum(p.stat().st_size for p in source.rglob("*") if p.is_file())
    print(f"[upload] {n_files} files totalling {total_size / 1e9:.2f} GB",
          flush=True)

    if dry_run:
        print("[upload] --dry-run: skipping actual transfer", flush=True)
        return

    print(f"[upload] starting transfer; resumable; safe to Ctrl-C and rerun",
          flush=True)
    t0 = time.time()
    api.upload_folder(
        folder_path=str(source),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="upload cached artifacts for paper replication",
        ignore_patterns=[
            "__pycache__/**", "*.pyc", "*.pyo",
            "*.tmp", ".DS_Store", "*.log",
            # Skip legacy / superseded experiment directories.
            "bar_p_joint_v1_loosestart/**",
            "cross_seed_replication/**",
            "universality/**",
            "v2_*.md",
        ],
    )
    print(f"[upload] done in {(time.time() - t0) / 60:.1f} min", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True,
                   help="local directory whose contents will be uploaded")
    p.add_argument("--repo-id", default="Jordanfmccann/polymorphism-is-rotation",
                   help="HF Hub dataset repo id")
    p.add_argument("--dry-run", action="store_true",
                   help="enumerate files and exit without uploading")
    args = p.parse_args()

    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.dry_run:
        print("HF_TOKEN env var not set; refusing to upload without it",
              flush=True)
        return 2

    upload(source=args.source, repo_id=args.repo_id, token=token,
           dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
