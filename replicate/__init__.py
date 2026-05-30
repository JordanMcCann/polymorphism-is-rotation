"""Replication orchestration for the polymorphism-is-rotation paper.

Public entry points:
    python -m replicate fetch-artifacts   # download cached Pythia + Dyck-3 artifacts
    python -m replicate run-fast          # analysis-only path (~30 min on RTX 2060)
    python -m replicate run-full          # full retrain + analysis (~14 h)
    python -m replicate figures           # regenerate the four paper figures
    python -m replicate verify            # compare outputs against paper claims
    python -m replicate test              # run the unit-test suite

All commands print expected wall time before running and tee results to
artifacts/. Outputs are deterministic up to BLAS/CUDA non-determinism noise
(see docs/REPLICATION.md for details).
"""

__version__ = "1.0.0"
