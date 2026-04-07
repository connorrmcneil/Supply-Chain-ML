"""
Build ensemble predictions CSV from GNN and Random Forest outputs.

Ensemble formula (from 5-fold CV, best alpha=0.4):
    ensemble_prob = 0.6 * rf_prob + 0.4 * gnn_prob
    ensemble_pred = 1 if ensemble_prob >= 0.5 else 0

Usage:
    python src/build_ensemble_predictions.py
"""

import pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent

gnn = pd.read_csv(ROOT / "outputs" / "gnn_predictions.csv")
rf = pd.read_csv(ROOT / "outputs" / "baseline_predictions.csv")

merged = gnn[["node_id", "gnn_pred", "gnn_prob"]].merge(
    rf[["node_id", "random_forest_prob"]], on="node_id", how="inner"
)

merged = merged.rename(columns={"random_forest_prob": "rf_prob"})
merged["ensemble_prob"] = 0.6 * merged["rf_prob"] + 0.4 * merged["gnn_prob"]
merged["ensemble_pred"] = (merged["ensemble_prob"] >= 0.5).astype(int)

out_path = ROOT / "outputs" / "ensemble_predictions.csv"
merged[["node_id", "rf_prob", "gnn_prob", "ensemble_prob", "ensemble_pred"]].to_csv(
    out_path, index=False
)

print(f"Wrote {len(merged)} rows to {out_path}")
