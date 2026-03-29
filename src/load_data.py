"""
load_data.py – Merge binary labels onto the feature table, produce master_nodes.csv.

Why two files?
  nodes_no_notes.csv  → clean feature source (no notes column).
  nodes_with_notes.csv → supplies only node_id + label.

We merge them so the rest of the pipeline uses one table.

Run:  python src/load_data.py
"""

import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "labelled_data"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"


def load_and_merge() -> pd.DataFrame:
    """Return a single master DataFrame with features + encoded labels."""
    nodes = pd.read_csv(DATA_DIR / "nodes_no_notes.csv")
    labels = pd.read_csv(
        DATA_DIR / "nodes_with_notes.csv", usecols=["node_id", "label"]
    )

    master = nodes.merge(labels, on="node_id", how="left")
    assert len(master) == len(nodes), (
        f"Row count changed after merge: {len(nodes)} → {len(master)}"
    )

    master["label_encoded"] = (
        master["label"]
        .map({"critical": 1, "non-critical": 0})
        .fillna(-1)
        .astype(int)
    )
    return master


def save_master(master: pd.DataFrame) -> pathlib.Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "master_nodes.csv"
    master.to_csv(out_path, index=False)
    print(f"Saved master node table ({len(master)} rows) → {out_path}")
    return out_path


if __name__ == "__main__":
    master = load_and_merge()
    save_master(master)
    print("\n--- Master table head ---")
    print(master.head(10).to_string())
    print(f"\nlabel_encoded value counts:\n"
          f"{master['label_encoded'].value_counts().sort_index().to_string()}")
