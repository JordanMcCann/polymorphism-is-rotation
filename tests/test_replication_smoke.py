"""Replication smoke tests.

These tests exercise the orchestration layer (`replicate/`) without needing
the downloaded artifacts. They catch:

  - CLI dispatch regressions
  - Verify-claim logic regressions
  - Layout/symlink regressions

For an end-to-end test with real artifacts, see `docs/REPLICATION.md`.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_help_runs() -> None:
    """`python -m replicate help` should exit 0 and mention every command."""
    result = subprocess.run(
        [sys.executable, "-m", "replicate", "help"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    out = result.stdout.lower()
    for cmd in ["fetch-artifacts", "run-fast", "run-full", "figures", "verify", "test"]:
        assert cmd in out, f"command '{cmd}' missing from help output"


def test_cli_unknown_command_exits_nonzero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "replicate", "definitely-not-a-command"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "unknown command" in result.stdout.lower()


def test_paths_ensure_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ensure_layout creates artifacts/ and the legacy symlink."""
    # Redirect REPO_ROOT for this test.
    from replicate import _paths
    monkeypatch.setattr(_paths, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_paths, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(_paths, "LEGACY_EXPERIMENTS", tmp_path / "experiments")

    _paths.ensure_layout()

    assert (tmp_path / "artifacts").is_dir()
    assert (tmp_path / "artifacts" / "scale" / "cache").is_dir()
    assert (tmp_path / "artifacts" / "seeds").is_dir()
    # Symlink exists and points to artifacts
    assert (tmp_path / "experiments").exists()


def test_verify_with_synthetic_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify passes when the JSON we expect contains the paper values."""
    from replicate import verify as verify_mod

    # Write a synthetic eigenvalue_spectrum.json that matches the paper.
    target = tmp_path / "scale" / "pythia_rotation" / "eigenvalue_spectrum.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({
        "summary": {
            "ks_stat_pooled_vs_haar": 0.0027,
            "ks_pvalue_pooled_vs_haar": 1.0,
            "pooled_mean_cos_theta": 0.0006,
            "mean_perm_dist": 29.6,
            "n_pair_site_combinations": 56,
            "n_pooled_eigenvalues": 28672,
        }
    }))

    monkeypatch.setattr(verify_mod, "ARTIFACTS", tmp_path)
    # Rebuild claims with the new ARTIFACTS path.
    new_claims = []
    for c in verify_mod.CLAIMS:
        new_path = tmp_path / c.json_path.relative_to(c.json_path.parents[2])
        new_claims.append(verify_mod.Claim(c.label, new_path, c.json_query, c.paper_value, c.rel_tol))

    rc = verify_mod.verify(new_claims)
    assert rc == 0, "verify should return 0 when every claim matches paper exactly"


def test_verify_detects_missing_jsons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify fails (exit 1) and reports MISSING when artifacts are not present."""
    from replicate import verify as verify_mod

    monkeypatch.setattr(verify_mod, "ARTIFACTS", tmp_path)
    new_claims = []
    for c in verify_mod.CLAIMS:
        new_path = tmp_path / c.json_path.relative_to(c.json_path.parents[2])
        new_claims.append(verify_mod.Claim(c.label, new_path, c.json_query, c.paper_value, c.rel_tol))

    rc = verify_mod.verify(new_claims)
    assert rc == 1, "verify should return 1 when artifacts are missing"


def test_verify_detects_value_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify rejects a value outside the tolerance window."""
    from replicate import verify as verify_mod

    target = tmp_path / "scale" / "pythia_rotation" / "eigenvalue_spectrum.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    # Wildly wrong KS stat (10x paper claim, way outside 20% tolerance).
    target.write_text(json.dumps({
        "summary": {
            "ks_stat_pooled_vs_haar": 0.027,
            "ks_pvalue_pooled_vs_haar": 1.0,
            "pooled_mean_cos_theta": 0.0006,
            "mean_perm_dist": 29.6,
            "n_pair_site_combinations": 56,
            "n_pooled_eigenvalues": 28672,
        }
    }))

    monkeypatch.setattr(verify_mod, "ARTIFACTS", tmp_path)
    new_claims = []
    for c in verify_mod.CLAIMS:
        new_path = tmp_path / c.json_path.relative_to(c.json_path.parents[2])
        new_claims.append(verify_mod.Claim(c.label, new_path, c.json_query, c.paper_value, c.rel_tol))

    rc = verify_mod.verify(new_claims)
    assert rc == 1, "verify should reject a value 10x off paper"
