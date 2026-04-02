"""
build_pyg_data.py – Assemble and validate a PyTorch Geometric Data object.

This is the orchestrator script.  It calls the helpers defined in the
other src/ modules, stitches everything into a single PyG Data object,
runs validation checks, and writes the final artefacts:

  data/processed/graph_data.pt   – the serialised Data object
  outputs/data_summary.json      – compact diagnostics summary

Run:  python src/build_pyg_data.py
"""

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# Allow imports from the same src/ directory when running as a script.
_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from build_features import build_feature_matrix, save_feature_columns  # noqa: E402
from build_graph import build_edge_index, build_node_mapping, save_node_mapping  # noqa: E402
from load_data import load_and_merge, save_master  # noqa: E402

DATA_DIR = _SRC_DIR.parent / "data" / "labelled_data"
PROCESSED_DIR = _SRC_DIR.parent / "data" / "processed"
OUTPUTS_DIR = _SRC_DIR.parent / "outputs"

INFRA_TYPES = {"PORT", "PLANT", "WAREHOUSE", "DC"}


# ── Mask creation ────────────────────────────────────────────────────────────

def create_masks(y: torch.Tensor, infra_mask: np.ndarray, seed: int = 42):
    """70 / 15 / 15 stratified split over infrastructure nodes only."""
    infra_idx = np.where(infra_mask)[0]
    infra_labels = y[infra_idx].numpy()

    train_idx, temp_idx = train_test_split(
        infra_idx, test_size=0.30, stratify=infra_labels, random_state=seed,
    )
    temp_labels = y[temp_idx].numpy()
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed,
    )

    n = len(y)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


# ── Validation ───────────────────────────────────────────────────────────────

def validate_and_print(data: Data, master: pd.DataFrame):
    sep = "=" * 60
    print(f"\n{sep}")
    print("PyG DATA OBJECT – SUMMARY & VALIDATION")
    print(sep)

    print(f"\n  num_nodes         : {data.num_nodes}")
    print(f"  num_edges         : {data.num_edges}")
    print(f"  num_node_features : {data.num_node_features}")
    print(f"  x shape           : {list(data.x.shape)}")
    print(f"  edge_index shape  : {list(data.edge_index.shape)}")
    print(f"  y shape           : {list(data.y.shape)}")

    tr = data.train_mask.sum().item()
    va = data.val_mask.sum().item()
    te = data.test_mask.sum().item()
    total = tr + va + te
    infra_count = master["node_type"].isin(INFRA_TYPES).sum()

    print(f"\n  train  : {tr}")
    print(f"  val    : {va}")
    print(f"  test   : {te}")
    print(f"  total  : {total}")

    assert total == infra_count, (
        f"Mask total ({total}) != infra node count ({infra_count})"
    )
    print(f"  masks cover all {infra_count} infrastructure nodes : OK")

    overlap = (
        (data.train_mask & data.val_mask).sum()
        + (data.train_mask & data.test_mask).sum()
        + (data.val_mask & data.test_mask).sum()
    )
    assert overlap == 0, f"Mask overlap detected ({overlap.item()} nodes)"
    print("  no mask overlap : OK")

    print("\n  --- Class balance per split ---")
    for name, mask in [("train", data.train_mask),
                       ("val", data.val_mask),
                       ("test", data.test_mask)]:
        labels = data.y[mask]
        n_crit = (labels == 1).sum().item()
        n_non = (labels == 0).sum().item()
        count = mask.sum().item()
        assert n_crit > 0 and n_non > 0, f"{name} split is missing a class!"
        print(f"  {name:5s} : critical={n_crit}  non-critical={n_non}  "
              f"total={count}  critical%={n_crit / count:.1%}")

    print(f"\n{sep}")
    print("ALL VALIDATION CHECKS PASSED")
    print(sep)


# ── Diagnostics summary ─────────────────────────────────────────────────────

def build_summary(data: Data, master: pd.DataFrame) -> dict:
    summary = {
        "node_count": int(data.num_nodes),
        "edge_count": int(data.num_edges),
        "feature_count": int(data.num_node_features),
        "label_counts_overall": {},
        "label_counts_by_node_type": {},
        "splits": {},
    }

    for val, name in [(1, "critical"), (0, "non-critical"), (-1, "unlabelled")]:
        summary["label_counts_overall"][name] = int((data.y == val).sum())

    for nt in sorted(master["node_type"].unique()):
        nt_mask = torch.tensor((master["node_type"] == nt).values, dtype=torch.bool)
        nt_labels = data.y[nt_mask]
        counts = {}
        for val, name in [(1, "critical"), (0, "non-critical"), (-1, "unlabelled")]:
            c = int((nt_labels == val).sum())
            if c > 0:
                counts[name] = c
        summary["label_counts_by_node_type"][nt] = counts

    for name, mask in [("train", data.train_mask),
                       ("val", data.val_mask),
                       ("test", data.test_mask)]:
        labels = data.y[mask]
        summary["splits"][name] = {
            "total": int(mask.sum()),
            "critical": int((labels == 1).sum()),
            "non-critical": int((labels == 0).sum()),
        }

    return summary


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    # Step 2 – load & merge
    print("Step 2: Loading and merging data …")
    master = load_and_merge()
    save_master(master)

    # Step 3 – feature engineering (including structural features from edges)
    print("\nStep 3: Building feature matrix …")
    edges = pd.read_csv(DATA_DIR / "edges.csv")
    feature_matrix, feature_cols = build_feature_matrix(master, edges=edges)
    save_feature_columns(feature_cols)
    print(f"  Feature matrix shape: {feature_matrix.shape}")

    # Step 4 – graph structure
    print("\nStep 4: Building graph structure …")
    node_id_to_idx = build_node_mapping(master)
    save_node_mapping(node_id_to_idx)
    edge_index = build_edge_index(edges, node_id_to_idx)
    print(f"  edge_index shape: {list(edge_index.shape)}")

    # Step 5 – assemble PyG Data
    print("\nStep 5: Assembling PyG Data object …")
    x = torch.tensor(feature_matrix, dtype=torch.float)
    y = torch.tensor(master["label_encoded"].values, dtype=torch.long)

    infra_mask = master["node_type"].isin(INFRA_TYPES).values
    train_mask, val_mask, test_mask = create_masks(y, infra_mask)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    # Validate
    validate_and_print(data, master)

    # Save Data object
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pt_path = PROCESSED_DIR / "graph_data.pt"
    torch.save(data, pt_path)
    print(f"\nSaved PyG Data → {pt_path}")

    # Save diagnostics summary
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary(data, master)
    summary_path = OUTPUTS_DIR / "data_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved diagnostics  → {summary_path}")


if __name__ == "__main__":
    main()
