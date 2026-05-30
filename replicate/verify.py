"""Verify that local outputs match the numerical claims in the paper.

Loads JSONs under `artifacts/` and compares to the values reported in the
paper. Tolerances are documented inline. A non-zero exit means at least one
claim failed; the script prints a table with the offending row(s).

The tolerances are deliberately tight enough to catch a meaningfully broken
replication and loose enough to absorb BLAS/CUDA non-determinism noise.
Tighter assertions are in the unit-test suite.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from ._paths import ARTIFACTS


@dataclass
class Claim:
    label: str
    json_path: Path
    json_query: str  # dotted path into the JSON
    paper_value: float
    rel_tol: float = 0.05  # default 5% relative tolerance

    def actual(self) -> float | None:
        if not self.json_path.exists():
            return None
        with open(self.json_path) as f:
            d = json.load(f)
        cur = d
        for key in self.json_query.split("."):
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                return None
        return float(cur)

    def check(self) -> tuple[bool, str]:
        a = self.actual()
        if a is None:
            return False, f"MISSING (no value at {self.json_query} in {self.json_path.name})"
        ok = abs(a - self.paper_value) <= self.rel_tol * max(abs(self.paper_value), 1e-12)
        return ok, f"actual={a:.6g}  paper={self.paper_value:.6g}  rel_err={abs(a - self.paper_value) / max(abs(self.paper_value), 1e-12):.3%}"


CLAIMS = [
    # Section 8.3, Figure 2 right panel -- the KS test
    Claim(
        label="KS statistic (pooled R eigenvalues vs Haar SO(512))",
        json_path=ARTIFACTS / "scale" / "pythia_rotation" / "eigenvalue_spectrum.json",
        json_query="summary.ks_stat_pooled_vs_haar",
        paper_value=0.0027,
        rel_tol=0.20,
    ),
    Claim(
        label="KS p-value (pooled, p ~= 1.000)",
        json_path=ARTIFACTS / "scale" / "pythia_rotation" / "eigenvalue_spectrum.json",
        json_query="summary.ks_pvalue_pooled_vs_haar",
        paper_value=1.000,
        rel_tol=0.01,
    ),
    Claim(
        label="Pooled mean cos(theta) ~= Haar 0.0006",
        json_path=ARTIFACTS / "scale" / "pythia_rotation" / "eigenvalue_spectrum.json",
        json_query="summary.pooled_mean_cos_theta",
        paper_value=0.0006,
        rel_tol=2.0,  # noisy at this magnitude; tolerance is generous
    ),
    Claim(
        label="||R - best_perm||_F mean (paper: 29.6)",
        json_path=ARTIFACTS / "scale" / "pythia_rotation" / "eigenvalue_spectrum.json",
        json_query="summary.mean_perm_dist",
        paper_value=29.6,
        rel_tol=0.02,
    ),
    Claim(
        label="Pair-site count = 56",
        json_path=ARTIFACTS / "scale" / "pythia_rotation" / "eigenvalue_spectrum.json",
        json_query="summary.n_pair_site_combinations",
        paper_value=56,
        rel_tol=0,
    ),
    Claim(
        label="Pooled eigenvalue count = 28,672",
        json_path=ARTIFACTS / "scale" / "pythia_rotation" / "eigenvalue_spectrum.json",
        json_query="summary.n_pooled_eigenvalues",
        paper_value=28672,
        rel_tol=0,
    ),
]


def verify(claims: list[Claim] = CLAIMS) -> int:
    print(f"{'#':>2}  {'STATUS':<7}  {'CLAIM':<60}  DETAIL", flush=True)
    print("-" * 130, flush=True)
    n_fail = 0
    for i, c in enumerate(claims, 1):
        ok, detail = c.check()
        status = "PASS" if ok else "FAIL"
        marker = "  " if ok else " *"
        print(f"{i:>2}{marker} {status:<7}  {c.label:<60}  {detail}", flush=True)
        if not ok:
            n_fail += 1

    print("-" * 130, flush=True)
    print(f"{len(claims) - n_fail}/{len(claims)} claims verified.", flush=True)
    if n_fail:
        print(f"\n{n_fail} claim(s) failed. See docs/PAPER_NUMBERS.md for which "
              f"script produces each value and how to debug.", flush=True)
    return 1 if n_fail else 0


def main() -> int:
    return verify()


if __name__ == "__main__":
    sys.exit(main())
