"""Download cached artifacts from the Hugging Face Hub.

We host two artifact sets on HF Hub at
`Jordanfmccann/polymorphism-is-rotation`:

  * Pythia-70m activation cache + SAE checkpoints (~9 GB total)
  * Dyck-3 trained seeds (0-4, 100-104) with lens outputs (~2 GB)

`fetch` is idempotent and resumable. Re-running skips files already present
with matching size. Use `--clean` to force redownload.
"""
from __future__ import annotations

import argparse
import sys
import time

from huggingface_hub import snapshot_download

from ._paths import ARTIFACTS, ensure_layout

HF_DATASET = "Jordanfmccann/polymorphism-is-rotation"


def fetch(clean: bool = False, allow_patterns: list[str] | None = None) -> None:
    """Download artifacts into `artifacts/`. Idempotent."""
    ensure_layout()
    print(f"[fetch] downloading {HF_DATASET} -> {ARTIFACTS} ...", flush=True)
    print("[fetch] expected size ~11 GB; resumable; safe to Ctrl-C and rerun", flush=True)
    t0 = time.time()
    snapshot_download(
        repo_id=HF_DATASET,
        repo_type="dataset",
        local_dir=str(ARTIFACTS),
        allow_patterns=allow_patterns,
        force_download=clean,
    )
    print(f"[fetch] done in {time.time() - t0:.1f}s", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", action="store_true", help="force redownload of every file")
    p.add_argument(
        "--only",
        choices=["all", "pythia", "dyck"],
        default="all",
        help="restrict download to one artifact subset",
    )
    args = p.parse_args()

    patterns = None
    if args.only == "pythia":
        patterns = ["scale/**"]
    elif args.only == "dyck":
        patterns = ["seeds/**", "cross_seed/**", "cross_checkpoint/**",
                    "independent_init/**", "bar_p_joint/**", "shared_io_init.pt"]

    fetch(clean=args.clean, allow_patterns=patterns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
