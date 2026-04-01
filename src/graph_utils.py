"""
graph_utils.py – Lightweight graph helpers for the Streamlit dashboard.

All functions work on plain Python dicts built from edges.csv.
No PyTorch / PyG dependency required at runtime.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import pandas as pd

AdjList = Dict[str, List[Tuple[str, str]]]  # node_id -> [(neighbour, rel_type)]

INFRA_TYPES = {"PORT", "PLANT", "WAREHOUSE", "DC"}


def _node_type(node_id: str) -> str:
    prefix = node_id.split("_")[0]
    return {"WH": "WAREHOUSE", "PROD": "PRODUCT", "ING": "INGREDIENT"}.get(prefix, prefix)


def build_adjacency(edges: pd.DataFrame) -> Tuple[AdjList, AdjList]:
    """Return (downstream, upstream) adjacency lists from edges.csv."""
    downstream: AdjList = defaultdict(list)
    upstream: AdjList = defaultdict(list)
    for _, row in edges.iterrows():
        src, dst, rel = row["src_id"], row["dst_id"], row["rel_type"]
        downstream[src].append((dst, rel))
        upstream[dst].append((src, rel))
    return dict(downstream), dict(upstream)


def bfs_downstream(node_id: str, downstream_adj: AdjList,
                   infra_only: bool = False) -> Set[str]:
    """BFS following forward edges. Returns all reachable node IDs (excluding start)."""
    visited: Set[str] = set()
    queue = deque([node_id])
    while queue:
        current = queue.popleft()
        for neighbour, _ in downstream_adj.get(current, []):
            if neighbour in visited:
                continue
            if infra_only and _node_type(neighbour) not in INFRA_TYPES:
                continue
            visited.add(neighbour)
            queue.append(neighbour)
    return visited


def blast_radius(node_id: str, downstream_adj: AdjList) -> dict:
    """Compute downstream blast radius grouped by node type."""
    reachable = bfs_downstream(node_id, downstream_adj, infra_only=True)
    by_type: Dict[str, List[str]] = defaultdict(list)
    for nid in reachable:
        by_type[_node_type(nid)].append(nid)

    return {
        "reachable_infra": len(reachable),
        "by_type": dict(by_type),
        "dc_count": len(by_type.get("DC", [])),
        "dc_list": sorted(by_type.get("DC", [])),
    }


def product_exposure(node_id: str, upstream_adj: AdjList,
                     downstream_adj: AdjList) -> dict:
    """Estimate which PRODUCTs are exposed if *node_id* fails.

    Trace: node -> downstream PLANTs (if not already a PLANT)
           PLANT -> upstream SUPPLIES -> INGREDIENTs
           ING  -> upstream REQUIRES -> PRODUCTs
    """
    nt = _node_type(node_id)

    # Collect relevant PLANTs
    if nt == "PLANT":
        plants = {node_id}
    elif nt == "PORT":
        plants = {n for n, r in downstream_adj.get(node_id, []) if r == "IMPORTS_TO"}
    elif nt == "WAREHOUSE":
        plants = {n for n, r in upstream_adj.get(node_id, []) if r == "SHIPS_TO"}
    elif nt == "DC":
        warehouses = {n for n, r in upstream_adj.get(node_id, []) if r == "REPLENISHES"}
        plants = set()
        for wh in warehouses:
            plants |= {n for n, r in upstream_adj.get(wh, []) if r == "SHIPS_TO"}
    else:
        return {"products": [], "product_count": 0}

    # PLANT -> ingredients that SUPPLY it -> products that REQUIRE those ingredients
    ingredients: Set[str] = set()
    for pl in plants:
        ingredients |= {n for n, r in upstream_adj.get(pl, []) if r == "SUPPLIES"}

    products: Set[str] = set()
    for ing in ingredients:
        products |= {n for n, r in upstream_adj.get(ing, []) if r == "REQUIRES"}

    return {
        "products": sorted(products),
        "product_count": len(products),
    }


def direct_neighbours(node_id: str, downstream_adj: AdjList,
                      upstream_adj: AdjList) -> Tuple[List[Tuple[str, str]],
                                                       List[Tuple[str, str]]]:
    """Return (downstream_list, upstream_list) of (node_id, rel_type) tuples."""
    return (
        downstream_adj.get(node_id, []),
        upstream_adj.get(node_id, []),
    )


def ego_graph(node_id: str, downstream_adj: AdjList,
              upstream_adj: AdjList, hops: int = 1
              ) -> Tuple[Set[str], List[Tuple[str, str, str]]]:
    """Return (nodes_set, edges_list) for a local ego graph.

    BFS outward in both directions up to *hops* layers.
    ``edges_list`` items are ``(src, dst, rel_type)`` tuples.
    """
    nodes: Set[str] = {node_id}
    frontier = {node_id}

    for _ in range(hops):
        next_frontier: Set[str] = set()
        for nid in frontier:
            for neighbour, _ in downstream_adj.get(nid, []):
                if neighbour not in nodes:
                    next_frontier.add(neighbour)
            for neighbour, _ in upstream_adj.get(nid, []):
                if neighbour not in nodes:
                    next_frontier.add(neighbour)
        nodes |= next_frontier
        frontier = next_frontier

    edge_list: List[Tuple[str, str, str]] = []
    seen_edges: Set[Tuple[str, str]] = set()
    for nid in nodes:
        for neighbour, rel in downstream_adj.get(nid, []):
            if neighbour in nodes and (nid, neighbour) not in seen_edges:
                edge_list.append((nid, neighbour, rel))
                seen_edges.add((nid, neighbour))

    return nodes, edge_list
