"""CLI dispatch for the replicate package.

Usage:
    python -m replicate <command> [args...]

Commands:
    fetch-artifacts   Download cached Pythia + Dyck-3 artifacts from HF Hub (~11 GB)
    run-fast          Analysis-only replication using cached artifacts (~30 min)
    run-full          Full retrain + analysis from scratch (~14 h)
    figures           Regenerate the four paper figures from cached outputs
    verify            Compare local outputs against paper numerical claims
    test              Run the unit-test suite (`pytest tests/`)
    help              Show this message
"""
from __future__ import annotations

import sys


def _help() -> int:
    print(__doc__, flush=True)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("help", "--help", "-h"):
        return _help()

    cmd, rest = argv[0], argv[1:]
    sys.argv = [f"replicate.{cmd}"] + rest

    if cmd == "fetch-artifacts":
        from . import fetch
        return fetch.main()
    if cmd == "run-fast":
        from . import run_fast
        return run_fast.main()
    if cmd == "run-full":
        from . import run_full
        return run_full.main()
    if cmd == "figures":
        from . import figures
        return figures.main()
    if cmd == "verify":
        from . import verify
        return verify.main()
    if cmd == "test":
        import subprocess
        return subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"])

    print(f"unknown command: {cmd}\n", flush=True)
    return _help() or 2


if __name__ == "__main__":
    sys.exit(main())
