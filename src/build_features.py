"""
build_features.py – Engineer model-ready features from master_nodes.csv.

Feature encoding rules
  node_type     → one-hot  (6 columns)
  region        → one-hot  (dynamic from CSV; NaN → "NONE")
  backup_level  → one-hot  (dynamic from CSV; NaN → "NONE")
  capacity_units, throughput_units, recovery_days,
    criticality_weight, volume_weight
                → MinMaxScaler fitted only on rows where the value exists
  is_backup_capable, substitutable
                → binary, kept as-is (NaN → 0)

When an edges DataFrame is provided, 7 graph-derived structural features
are appended (in_degree … capacity_utilization), computed by
build_structural_features.py.

Run:  python src/build_features.py
"""

import json
import pathlib
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

PROCESSED_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs"

CONTINUOUS_COLS = [
    "capacity_units",
    "throughput_units",
    "recovery_days",
    "criticality_weight",
    "volume_weight",
]
BINARY_COLS = ["is_backup_capable", "substitutable"]


def build_feature_matrix(
    master: pd.DataFrame,
    edges: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return (feature_matrix [N, F], feature_column_names).

    When *edges* is supplied the 7 graph-derived structural features from
    ``build_structural_features`` are appended, expanding the feature
    dimension from 26 to 33.
    """
    df = master.copy()

    # ── One-hot: node_type ───────────────────────────────────────────────
    node_type_dummies = pd.get_dummies(df["node_type"], prefix="node_type")

    # ── One-hot: region (NaN → "NONE" placeholder) ──────────────────────
    df["region_clean"] = df["region"].fillna("NONE")
    region_dummies = pd.get_dummies(df["region_clean"], prefix="region")

    # ── One-hot: backup_level (NaN → "NONE" placeholder) ────────────────
    df["backup_level_clean"] = df["backup_level"].fillna("NONE")
    backup_dummies = pd.get_dummies(df["backup_level_clean"], prefix="backup_level")

    # ── Scale continuous columns ─────────────────────────────────────────
    scaled_series = {}
    for col in CONTINUOUS_COLS:
        present = df[col].notna()
        result = pd.Series(0.0, index=df.index, name=col)
        if present.sum() > 0:
            scaler = MinMaxScaler()
            result.loc[present] = scaler.fit_transform(
                df.loc[present, col].values.reshape(-1, 1)
            ).flatten()
        scaled_series[col] = result
    scaled_df = pd.DataFrame(scaled_series)

    # ── Binary columns (NaN → 0) ────────────────────────────────────────
    binary_df = df[BINARY_COLS].fillna(0).astype(float)

    # ── Assemble base features ───────────────────────────────────────────
    parts = [node_type_dummies, region_dummies, backup_dummies, scaled_df, binary_df]

    # ── Structural features (optional) ───────────────────────────────────
    if edges is not None:
        from build_structural_features import compute_structural_features
        struct_df, _ = compute_structural_features(master, edges)
        parts.append(struct_df)

    feature_df = pd.concat(parts, axis=1)
    feature_cols = list(feature_df.columns)
    feature_matrix = feature_df.values.astype(np.float32)

    return feature_matrix, feature_cols


def save_feature_columns(feature_cols: List[str]) -> pathlib.Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "feature_columns.json"
    with open(out_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved feature column names ({len(feature_cols)} features) → {out_path}")
    return out_path


if __name__ == "__main__":
    master = pd.read_csv(PROCESSED_DIR / "master_nodes.csv")
    feature_matrix, feature_cols = build_feature_matrix(master)
    save_feature_columns(feature_cols)
    print(f"\nFeature matrix shape : {feature_matrix.shape}")
    print(f"Feature columns      : {feature_cols}")
    print(f"\nFirst 5 rows:\n{feature_matrix[:5]}")
