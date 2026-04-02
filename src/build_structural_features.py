"""
build_structural_features.py – Compute graph-derived structural features
for every node using the edge list.

Features (7 columns, all MinMax-scaled):
  in_degree             – number of incoming edges
  out_degree            – number of outgoing edges
  total_degree          – in + out
  upstream_redundancy   – distinct upstream infrastructure nodes
  downstream_fanout     – distinct downstream infrastructure nodes
  reachable_dc_count    – DCs reachable via downstream BFS
  capacity_utilization  – throughput / capacity (0 if either is missing)
"""

import pathlib
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from graph_utils import INFRA_TYPES, _node_type, blast_radius, build_adjacency

DATA_DIR = _SRC_DIR.parent / "data" / "labelled_data"

STRUCTURAL_COLS = [
    "in_degree",
    "out_degree",
    "total_degree",
    "upstream_redundancy",
    "downstream_fanout",
    "reachable_dc_count",
    "capacity_utilization",
]


def compute_structural_features(
    master: pd.DataFrame,
    edges: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """Return a DataFrame (aligned to master's index) with 7 scaled structural features."""
    downstream_adj, upstream_adj = build_adjacency(edges)

    records = []
    for _, row in master.iterrows():
        nid = row["node_id"]

        ds_list = downstream_adj.get(nid, [])
        us_list = upstream_adj.get(nid, [])

        in_deg = len(us_list)
        out_deg = len(ds_list)

        us_infra = sum(1 for n, _ in us_list if _node_type(n) in INFRA_TYPES)
        ds_infra = sum(1 for n, _ in ds_list if _node_type(n) in INFRA_TYPES)

        br = blast_radius(nid, downstream_adj)

        cap = row.get("capacity_units")
        thr = row.get("throughput_units")
        if pd.notna(cap) and pd.notna(thr) and cap > 0:
            cap_util = thr / cap
        else:
            cap_util = 0.0

        records.append({
            "in_degree": in_deg,
            "out_degree": out_deg,
            "total_degree": in_deg + out_deg,
            "upstream_redundancy": us_infra,
            "downstream_fanout": ds_infra,
            "reachable_dc_count": br["dc_count"],
            "capacity_utilization": cap_util,
        })

    raw_df = pd.DataFrame(records, index=master.index)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(raw_df.values)
    scaled_df = pd.DataFrame(scaled, columns=STRUCTURAL_COLS, index=master.index)

    return scaled_df, STRUCTURAL_COLS


if __name__ == "__main__":
    master = pd.read_csv(
        _SRC_DIR.parent / "data" / "processed" / "master_nodes.csv"
    )
    edges = pd.read_csv(DATA_DIR / "edges.csv")
    df, cols = compute_structural_features(master, edges)
    print(f"Structural features shape: {df.shape}")
    print(f"Columns: {cols}")
    print(f"\nSample (first 5 rows):\n{df.head()}")
    print(f"\nDescribe:\n{df.describe()}")
