"""Build a small (~50 MB) tarball for GitHub Release attachment.

Contains only the artifacts a reviewer needs to inspect / re-verify the
Dyck-3 trained models without downloading the full ~11 GB HF dataset:

  - Final checkpoint of each seed (final step only, not every 500-step
    checkpoint)
  - shared_io_init.pt
  - All bar_outputs JSONs
  - Headline result JSONs from each experiment
  - A small subset of the Pythia rotation results (the JSONs, not the
    activation cache)

Usage:
    python scripts/build_github_release.py \\
        --source artifacts \\
        --out polymorphism-replication-lite.tar.gz
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


def find_final_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the highest-step ckpt_NNNNNNN.pt in `checkpoints_dir`."""
    if not checkpoints_dir.exists():
        return None
    candidates = sorted(checkpoints_dir.glob("ckpt_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    return candidates[-1] if candidates else None


def stage_files(source: Path, staging: Path) -> tuple[int, int]:
    """Copy a curated subset of `source` into `staging`. Return (n_files, bytes)."""
    n_files = 0
    n_bytes = 0

    # 1. Shared I/O init (small).
    sio = source / "shared_io_init.pt"
    if sio.exists():
        dst = staging / "shared_io_init.pt"
        shutil.copy2(sio, dst)
        n_files += 1
        n_bytes += dst.stat().st_size

    # 2. Per-seed: final checkpoint + bar_outputs + summary of lens_outputs.
    for seed_dir in sorted((source / "seeds").iterdir() if (source / "seeds").exists() else []):
        if not seed_dir.is_dir():
            continue
        seed_name = seed_dir.name
        dst_seed = staging / "seeds" / seed_name
        dst_seed.mkdir(parents=True, exist_ok=True)

        ckpt = find_final_checkpoint(seed_dir / "checkpoints")
        if ckpt:
            dst_ckpt = dst_seed / "checkpoints" / ckpt.name
            dst_ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ckpt, dst_ckpt)
            n_files += 1
            n_bytes += dst_ckpt.stat().st_size

        bo = seed_dir / "bar_outputs"
        if bo.exists():
            dst_bo = dst_seed / "bar_outputs"
            shutil.copytree(bo, dst_bo, dirs_exist_ok=True)
            for p in dst_bo.rglob("*"):
                if p.is_file():
                    n_files += 1
                    n_bytes += p.stat().st_size

        # Lens outputs: keep only JSONs, drop large .pt SAE caches.
        lo = seed_dir / "lens_outputs"
        if lo.exists():
            for json_file in lo.rglob("*.json"):
                rel = json_file.relative_to(lo)
                dst_lo = dst_seed / "lens_outputs" / rel
                dst_lo.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(json_file, dst_lo)
                n_files += 1
                n_bytes += dst_lo.stat().st_size

    # 3. Pythia rotation result JSONs (no activation cache).
    for sub in ["pythia_rotation"]:
        src_sub = source / "scale" / sub
        if src_sub.exists():
            for f in src_sub.rglob("*"):
                if f.is_file() and not f.name.endswith(".npy"):
                    rel = f.relative_to(src_sub)
                    dst_f = staging / "scale" / sub / rel
                    dst_f.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst_f)
                    n_files += 1
                    n_bytes += dst_f.stat().st_size

    # 4. Cross-seed / cross-checkpoint / independent-init / bar_p_joint result JSONs.
    for sub in ["cross_seed", "cross_checkpoint", "independent_init", "bar_p_joint"]:
        src_sub = source / sub
        if src_sub.exists():
            for f in src_sub.rglob("*"):
                if f.is_file() and f.suffix in (".json", ".md", ".pt"):
                    rel = f.relative_to(src_sub)
                    dst_f = staging / sub / rel
                    dst_f.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst_f)
                    n_files += 1
                    n_bytes += dst_f.stat().st_size

    return n_files, n_bytes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True,
                   help="path to the experiments/ directory to bundle")
    p.add_argument("--out", type=Path,
                   default=Path("polymorphism-replication-lite.tar.gz"),
                   help="output tarball path")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        print(f"[release] staging in {staging}", flush=True)
        n_files, n_bytes = stage_files(args.source, staging)
        print(f"[release] staged {n_files} files, {n_bytes / 1e6:.1f} MB", flush=True)

        print(f"[release] writing tarball to {args.out}", flush=True)
        with tarfile.open(args.out, "w:gz") as tf:
            for p in sorted(staging.rglob("*")):
                if p.is_file():
                    tf.add(p, arcname=str(p.relative_to(staging)))

        size = args.out.stat().st_size
        print(f"[release] {args.out}  ({size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
