import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit_r2_credibility import fit_predict_rf
from src.tree_modeling import COMMON_LEAKAGE_RISK_FEATURES, GROUP_COLUMN


ROOT = Path(__file__).resolve().parents[1]


def test_grouped_split_files_have_no_product_id_overlap():
    train_df = pd.read_csv(ROOT / "datasets/splits/train_grouped.csv", usecols=[GROUP_COLUMN])
    validation_df = pd.read_csv(ROOT / "datasets/splits/validation_grouped.csv", usecols=[GROUP_COLUMN])
    test_df = pd.read_csv(ROOT / "datasets/splits/test_grouped.csv", usecols=[GROUP_COLUMN])

    train_groups = set(train_df[GROUP_COLUMN])
    validation_groups = set(validation_df[GROUP_COLUMN])
    test_groups = set(test_df[GROUP_COLUMN])

    assert not train_groups & validation_groups
    assert not train_groups & test_groups
    assert not validation_groups & test_groups


def test_selected_random_forest_feature_set_excludes_declared_leakage_risk_columns():
    summary = json.loads(
        (ROOT / "artifacts/random_forest_tuning/best_tuning_summary.json").read_text(
            encoding="utf-8"
        )
    )

    selected_features = set(summary["feature_names"])

    assert not selected_features & COMMON_LEAKAGE_RISK_FEATURES


def test_shuffled_target_sanity_collapses_on_synthetic_signal():
    rng = np.random.default_rng(42)
    train_df = pd.DataFrame(
        {
            "product_id": np.arange(80),
            "signal": np.arange(80, dtype=float),
            "category": np.where(np.arange(80) % 2 == 0, "a", "b"),
        }
    )
    train_df["price"] = train_df["signal"] * 10.0
    validation_df = pd.DataFrame(
        {
            "product_id": np.arange(80, 120),
            "signal": np.arange(40, dtype=float),
            "category": np.where(np.arange(40) % 2 == 0, "a", "b"),
        }
    )
    validation_df["price"] = validation_df["signal"] * 10.0
    config = {
        "target_mode": "raw",
        "onehot_min_frequency": 1,
        "model_params": {
            "n_estimators": 30,
            "min_samples_leaf": 1,
            "max_features": 1.0,
            "random_state": 42,
            "n_jobs": 1,
        },
    }

    _, real_metrics = fit_predict_rf(train_df, validation_df, ["signal", "category"], config)
    shuffled_train_df = train_df.copy()
    shuffled_train_df["price"] = rng.permutation(shuffled_train_df["price"].to_numpy())
    _, shuffled_metrics = fit_predict_rf(
        shuffled_train_df,
        validation_df,
        ["signal", "category"],
        config,
    )

    assert shuffled_metrics["R2"] < real_metrics["R2"] - 0.5
