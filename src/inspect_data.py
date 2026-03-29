"""
inspect_data.py – Load all three source CSVs and run validation checks.

This is a pure diagnostics script.  It prints shapes, dtypes, null counts,
duplicate checks, value-count breakdowns, and verifies referential integrity
between the node tables and the edge table.  Nothing is written to disk.

Run:  python src/inspect_data.py
"""

import pathlib
import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "labelled_data"

INFRA_TYPES = {"PORT", "PLANT", "WAREHOUSE", "DC"}
CONTEXT_TYPES = {"INGREDIENT", "PRODUCT"}


def load_raw_csvs():
    """Return (nodes_no_notes, nodes_with_notes, edges) DataFrames."""
    nodes_no_notes = pd.read_csv(DATA_DIR / "nodes_no_notes.csv")
    nodes_with_notes = pd.read_csv(DATA_DIR / "nodes_with_notes.csv")
    edges = pd.read_csv(DATA_DIR / "edges.csv")
    return nodes_no_notes, nodes_with_notes, edges


def inspect(nodes_no_notes, nodes_with_notes, edges):
    """Run every sanity check and print results to stdout."""
    sep = "=" * 60

    print(sep)
    print("DATA INSPECTION & VALIDATION")
    print(sep)

    # ---- Shapes ----------------------------------------------------------
    print(f"\nnodes_no_notes   : {nodes_no_notes.shape[0]} rows  x  {nodes_no_notes.shape[1]} cols")
    print(f"nodes_with_notes : {nodes_with_notes.shape[0]} rows  x  {nodes_with_notes.shape[1]} cols")
    print(f"edges            : {edges.shape[0]} rows  x  {edges.shape[1]} cols")

    # ---- Dtypes ----------------------------------------------------------
    for name, df in [("nodes_no_notes", nodes_no_notes),
                     ("nodes_with_notes", nodes_with_notes),
                     ("edges", edges)]:
        print(f"\n--- {name} dtypes ---")
        print(df.dtypes.to_string())

    # ---- Null counts -----------------------------------------------------
    for name, df in [("nodes_no_notes", nodes_no_notes),
                     ("nodes_with_notes", nodes_with_notes),
                     ("edges", edges)]:
        print(f"\n--- {name} null counts ---")
        print(df.isnull().sum().to_string())

    # ---- Duplicate node IDs ----------------------------------------------
    dup_nn = nodes_no_notes["node_id"].duplicated().sum()
    dup_wn = nodes_with_notes["node_id"].duplicated().sum()
    print(f"\nDuplicate node_ids in nodes_no_notes  : {dup_nn}")
    print(f"Duplicate node_ids in nodes_with_notes: {dup_wn}")
    assert dup_nn == 0, "Duplicate node_ids in nodes_no_notes!"
    assert dup_wn == 0, "Duplicate node_ids in nodes_with_notes!"

    # ---- node_id sets match across files ---------------------------------
    ids_nn = set(nodes_no_notes["node_id"])
    ids_wn = set(nodes_with_notes["node_id"])
    assert ids_nn == ids_wn, (
        f"node_id mismatch!  only in no_notes: {ids_nn - ids_wn}, "
        f"only in with_notes: {ids_wn - ids_nn}"
    )
    print("node_id sets match across both files : OK")

    # ---- Duplicate edges -------------------------------------------------
    dup_edges = edges.duplicated(subset=["src_id", "rel_type", "dst_id"]).sum()
    print(f"Duplicate edges (src, rel, dst)      : {dup_edges}")

    # ---- Node counts by node_type ----------------------------------------
    print("\n--- Node counts by node_type ---")
    print(nodes_no_notes["node_type"].value_counts().sort_index().to_string())

    # ---- Label counts overall --------------------------------------------
    print("\n--- Label counts (overall) ---")
    print(nodes_with_notes["label"].value_counts(dropna=False).to_string())

    # ---- Label counts by node_type ---------------------------------------
    print("\n--- Label counts by node_type ---")
    for nt in sorted(nodes_with_notes["node_type"].unique()):
        subset = nodes_with_notes[nodes_with_notes["node_type"] == nt]
        counts = subset["label"].value_counts(dropna=False).to_dict()
        print(f"  {nt:12s}: {counts}")

    # ---- Context nodes must have no labels -------------------------------
    context_labels = nodes_with_notes[
        nodes_with_notes["node_type"].isin(CONTEXT_TYPES)
    ]["label"].dropna()
    assert len(context_labels) == 0, (
        f"Context nodes should have no labels, found {len(context_labels)}"
    )
    print("\nContext nodes have no labels : OK")

    # ---- Edge counts by rel_type -----------------------------------------
    print("\n--- Edge counts by rel_type ---")
    print(edges["rel_type"].value_counts().sort_index().to_string())

    # ---- Edge endpoint integrity -----------------------------------------
    all_node_ids = set(nodes_no_notes["node_id"])
    src_missing = set(edges["src_id"]) - all_node_ids
    dst_missing = set(edges["dst_id"]) - all_node_ids
    assert len(src_missing) == 0, f"Unknown src_ids: {src_missing}"
    assert len(dst_missing) == 0, f"Unknown dst_ids: {dst_missing}"
    print("All edge endpoints exist in node table : OK")

    print(f"\n{sep}")
    print("ALL VALIDATION CHECKS PASSED")
    print(sep)


if __name__ == "__main__":
    nn, wn, e = load_raw_csvs()
    inspect(nn, wn, e)
