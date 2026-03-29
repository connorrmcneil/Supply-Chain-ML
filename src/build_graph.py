"""
build_graph.py – Build node-ID-to-integer mapping and directed edge_index tensor.

The mapping preserves the row order of master_nodes.csv so that row i in
the feature matrix corresponds to node index i in the graph.

Run:  python src/build_graph.py
"""

import json
import pathlib

import pandas as pd
import torch

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "labelled_data"
PROCESSED_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "outputs"


def build_node_mapping(master: pd.DataFrame) -> dict:
    """Map each node_id string to an integer index (0-based, row order)."""
    node_id_to_idx = {nid: idx for idx, nid in enumerate(master["node_id"])}
    assert len(node_id_to_idx) == len(master), "Duplicate node_ids in master table"
    return node_id_to_idx


def build_edge_index(edges: pd.DataFrame, node_id_to_idx: dict) -> torch.Tensor:
    """Return a [2, num_edges] LongTensor of directed edges."""
    src = edges["src_id"].map(node_id_to_idx)
    dst = edges["dst_id"].map(node_id_to_idx)

    assert src.isna().sum() == 0, f"{src.isna().sum()} src_ids not in node mapping"
    assert dst.isna().sum() == 0, f"{dst.isna().sum()} dst_ids not in node mapping"

    edge_index = torch.tensor(
        [src.astype(int).tolist(), dst.astype(int).tolist()],
        dtype=torch.long,
    )

    num_nodes = len(node_id_to_idx)
    assert edge_index.min() >= 0, "Negative index in edge_index"
    assert edge_index.max() < num_nodes, (
        f"Index {edge_index.max().item()} out of range (num_nodes={num_nodes})"
    )
    return edge_index


def save_node_mapping(node_id_to_idx: dict) -> pathlib.Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "node_id_to_idx.json"
    with open(out_path, "w") as f:
        json.dump(node_id_to_idx, f, indent=2)
    print(f"Saved node mapping ({len(node_id_to_idx)} nodes) → {out_path}")
    return out_path


if __name__ == "__main__":
    master = pd.read_csv(PROCESSED_DIR / "master_nodes.csv")
    edges = pd.read_csv(DATA_DIR / "edges.csv")

    mapping = build_node_mapping(master)
    save_node_mapping(mapping)

    ei = build_edge_index(edges, mapping)
    print(f"\nedge_index shape : {list(ei.shape)}")
    print(f"edge_index[:, :5]:\n{ei[:, :5]}")
