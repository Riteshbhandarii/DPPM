#!/usr/bin/env python3
"""
Purpose:
Generate and freeze the final strict connected-component train/validation/test split
artifacts for the thesis evaluation protocol (issue #35).

Inputs:
- datasets/cleaned/clean_master_dataset.csv by default

Outputs:
- datasets/splits_strict/train_strict.csv
- datasets/splits_strict/validation_strict.csv
- datasets/splits_strict/test_strict.csv
- datasets/splits_strict/strict_group_assignment.csv
- datasets/splits_strict/strict_split_summary.json

Assumptions:
- This is the explicitly approved split-generation entrypoint required by
  docs/evaluation/02_PROTOCOL_SPECIFICATION.md. It reuses the exact component
  construction and assignment logic from the balance diagnostic
  (scripts/analyze_candidate_component_split_balance.py) so the frozen split is
  identical to the diagnostic assignment for the documented seed.
- The default seed is 32, selected by the candidate balance diagnostic.
- Split files contain only the original input columns; group/component metadata is
  saved separately in strict_group_assignment.csv.
- Existing split artifacts are never overwritten unless --force is passed, because
  the frozen split is a first-class thesis artifact.

How to run:
python3 scripts/generate_strict_split.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from analyze_candidate_component_split_balance import (
    SPLITS,
    TARGET_PROPORTIONS,
    assign_components,
    build_component_frame,
    component_summary,
    resolve_target_column,
    run_leakage_checks,
    split_row_summary,
    validate_columns,
)


DEFAULT_SEED = 32
SPLIT_FILE_NAMES = {
    "train": "train_strict.csv",
    "validation": "validation_strict.csv",
    "test": "test_strict.csv",
}
ASSIGNMENT_COLUMNS = [
    "row_index",
    "component_id",
    "component_size",
    "canonical_product_id",
    "identity_key",
    "split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and freeze the final strict connected-component split artifacts."
    )
    parser.add_argument(
        "--input-path",
        default="datasets/cleaned/clean_master_dataset.csv",
        help="Cleaned master dataset to split.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/splits_strict",
        help="Directory for the frozen strict split artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Assignment seed. Defaults to the documented diagnostic seed.",
    )
    parser.add_argument(
        "--target-column",
        default=None,
        help="Target price column. Defaults to price if present, otherwise listing_price.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing strict split artifacts. Off by default to protect the frozen split.",
    )
    return parser.parse_args()


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_existing_artifacts(output_dir: Path, force: bool) -> None:
    if force or not output_dir.exists():
        return
    existing = sorted(path.name for path in output_dir.iterdir())
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing strict split artifacts in "
            f"{output_dir.as_posix()}: {existing}. Pass --force to regenerate."
        )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    refuse_existing_artifacts(output_dir, args.force)

    frame = pd.read_csv(input_path)
    original_columns = list(frame.columns)
    target_column = resolve_target_column(frame, args.target_column)
    validate_columns(frame, target_column)

    component_frame = build_component_frame(frame, target_column)
    components = component_summary(component_frame)
    total_rows = len(component_frame)

    assignments = assign_components(components, args.seed, total_rows)
    rows = component_frame.merge(
        assignments[["component_id", "split"]], on="component_id", how="left"
    )
    if rows["split"].isna().any():
        raise AssertionError("Some rows did not receive a split assignment.")

    leakage_summary = run_leakage_checks(rows)
    row_summary = split_row_summary(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_in_input_order = rows.sort_values("row_index")
    for split in SPLITS:
        split_rows = rows_in_input_order[rows_in_input_order["split"] == split]
        split_rows[original_columns].to_csv(output_dir / SPLIT_FILE_NAMES[split], index=False)

    rows_in_input_order[ASSIGNMENT_COLUMNS].to_csv(
        output_dir / "strict_group_assignment.csv", index=False
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "generator": "scripts/generate_strict_split.py",
        "protocol": "docs/evaluation/02_PROTOCOL_SPECIFICATION.md",
        "input_path": input_path.as_posix(),
        "input_sha256": sha256_of_file(input_path),
        "target_column": target_column,
        "seed": args.seed,
        "target_proportions": TARGET_PROPORTIONS,
        "total_rows": total_rows,
        "total_components": int(len(components)),
        "largest_component_rows": int(components["component_size"].max()),
        "splits": {
            record["split"]: {
                "file": SPLIT_FILE_NAMES[record["split"]],
                "row_count": int(record["row_count"]),
                "row_pct": round(float(record["row_pct"]), 4),
                "component_count": int(record["component_count"]),
                "product_id_count": int(record["product_id_count"]),
                "identity_key_count": int(record["identity_key_count"]),
            }
            for record in row_summary.to_dict("records")
        },
        "leakage_checks": leakage_summary.to_dict("records"),
    }
    (output_dir / "strict_split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Frozen strict split artifacts written to: {output_dir.as_posix()}")
    for split in SPLITS:
        detail = summary["splits"][split]
        print(f"  {split}: {detail['row_count']} rows ({detail['row_pct']:.2f}%)")
    print("All leakage assertions passed.")


if __name__ == "__main__":
    main()
