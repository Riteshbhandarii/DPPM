"""Verify reproducibility WITHOUT touching any frozen artifact.

Two cheap, read-only checks:
  1. Regenerate the strict split into a temporary directory (input = the frozen
     clean_master_dataset.csv) and SHA256-compare against datasets/splits_strict/*.csv.
     Proves the split is deterministic (seed 32) and byte-identically reproducible.
  2. Confirm the frozen holdout headline (test_MAE ~= 69.46) is present and unchanged.

This never overwrites the frozen split, never re-runs model tuning, and never re-runs the
one-shot holdout. Runs in a few seconds. Exit code 0 = reproducible, 1 = mismatch.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

FROZEN_SPLIT_DIR = Path("datasets/splits_strict")
SPLIT_CSVS = [
    "train_strict.csv",
    "validation_strict.csv",
    "test_strict.csv",
    "strict_group_assignment.csv",
]
HOLDOUT_METRICS = Path("artifacts/strict_final_holdout/final_holdout_metrics.json")
EXPECTED_HOLDOUT_MAE = 69.46
HOLDOUT_TOL = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_split_reproduces() -> bool:
    print("[1/2] Regenerating the strict split into a temp dir (frozen split untouched)...")
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "scripts/generate_strict_split.py", "--output-dir", tmp],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print("  FAIL: split generation errored:\n" + result.stderr)
            return False
        ok = True
        for name in SPLIT_CSVS:
            frozen, regen = FROZEN_SPLIT_DIR / name, Path(tmp) / name
            if not frozen.exists() or not regen.exists():
                print(f"  {name}: MISSING (frozen={frozen.exists()}, regen={regen.exists()})")
                ok = False
                continue
            same = sha256(frozen) == sha256(regen)
            print(f"  {name}: {'identical' if same else 'DIFFERS'}")
            ok = ok and same
    return ok


def check_holdout_headline() -> bool:
    print("[2/2] Checking the frozen holdout headline...")
    if not HOLDOUT_METRICS.exists():
        print(f"  FAIL: {HOLDOUT_METRICS} missing")
        return False
    mae = float(json.loads(HOLDOUT_METRICS.read_text())["test_MAE"])
    ok = abs(mae - EXPECTED_HOLDOUT_MAE) <= HOLDOUT_TOL
    print(f"  test_MAE = {mae:.2f} € (expected ~{EXPECTED_HOLDOUT_MAE}) -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main() -> int:
    split_ok = check_split_reproduces()
    holdout_ok = check_holdout_headline()
    passed = split_ok and holdout_ok
    print("\nRESULT: " + (
        "PASS — strict split reproduces byte-identically; frozen holdout intact."
        if passed else "FAIL — see mismatches above."
    ))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
