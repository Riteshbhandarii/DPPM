"""Canonicalization and connected-component grouping for the strict evaluation protocol.

This module is the single source of truth for the strict identity rule defined in
docs/evaluation/02_PROTOCOL_SPECIFICATION.md. The functions are copied verbatim from
scripts/analyze_candidate_component_split_balance.py, which generated the frozen split
in datasets/splits_strict/. tests/test_strict_protocol.py asserts that this module
reproduces the frozen component assignment exactly, so the two implementations cannot
drift apart silently.

Used for:
- rebuilding connected components inside any subset of rows (e.g. the strict training
  split) so cross-validation folds can be grouped by the same rule as the frozen split.
"""

from __future__ import annotations

import math
import numbers
import re
import unicodedata

import pandas as pd


MISSING_SENTINEL = "__missing__"
WHITESPACE_RE = re.compile(r"\s+")
KEY_SEPARATOR = "\x1f"
IDENTITY_COLUMNS = ("part_name", "brand", "model", "year_start", "year_end")
COMPONENT_GROUP_COLUMN = "strict_component_group"


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def canonical_text(value: object) -> str:
    if pd.isna(value):
        return MISSING_SENTINEL

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if math.isfinite(numeric_value) and numeric_value.is_integer():
            text = str(int(numeric_value))
        else:
            text = str(value)
    else:
        text = str(value)

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = WHITESPACE_RE.sub(" ", text.strip())
    return text if text else MISSING_SENTINEL


def canonicalize_series(series: pd.Series) -> pd.Series:
    return series.map(canonical_text).astype("string")


def build_identity_key(frame: pd.DataFrame) -> pd.Series:
    """Return the canonical strict identity key for every row."""

    missing = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"Strict identity columns missing from data: {missing}")

    key = canonicalize_series(frame[IDENTITY_COLUMNS[0]])
    for column in IDENTITY_COLUMNS[1:]:
        key = key.str.cat(canonicalize_series(frame[column]), sep=KEY_SEPARATOR)
    return key


def _union_by_key(union_find: UnionFind, keys: pd.Series) -> None:
    first_seen: dict[str, int] = {}
    for row_position, key in enumerate(keys.astype(str)):
        first_row = first_seen.get(key)
        if first_row is None:
            first_seen[key] = row_position
        else:
            union_find.union(first_row, row_position)


def add_component_group(
    frame: pd.DataFrame,
    group_column: str = COMPONENT_GROUP_COLUMN,
) -> pd.DataFrame:
    """Attach the connected-component group id to every row of ``frame``.

    Rows are connected when they share the canonical ``product_id`` or the canonical
    strict identity key. Components are local to the given frame: the ids are only
    meaningful for grouping rows of this frame (e.g. CV folds inside the training
    split), not for matching against another frame.
    """

    if "product_id" not in frame.columns:
        raise KeyError("Column 'product_id' is required to build strict components.")

    output = frame.copy()
    canonical_product_id = canonicalize_series(frame["product_id"]).reset_index(drop=True)
    identity_key = build_identity_key(frame).reset_index(drop=True)

    union_find = UnionFind(len(output))
    _union_by_key(union_find, canonical_product_id)
    _union_by_key(union_find, identity_key)

    roots = [union_find.find(index) for index in range(len(output))]
    root_to_component = {root: f"component_{index:06d}" for index, root in enumerate(dict.fromkeys(roots), start=1)}
    output[group_column] = [root_to_component[root] for root in roots]
    return output
