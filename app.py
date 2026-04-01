"""
Supply-Chain Criticality Dashboard
===================================
Multi-tab Streamlit app showing predicted infrastructure criticality,
interactive graph views, blast-radius analysis, and model error review.

Run:  streamlit run app.py
"""

import pathlib
import sys

import pandas as pd
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

SRC_DIR = pathlib.Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from graph_utils import (
    _node_type,
    blast_radius,
    build_adjacency,
    direct_neighbours,
    ego_graph,
    product_exposure,
)

ROOT = pathlib.Path(__file__).resolve().parent

NODE_TYPE_COLORS = {
    "PORT": "#4a90d9",
    "PLANT": "#e8943a",
    "WAREHOUSE": "#50b87a",
    "DC": "#9b59b6",
    "PRODUCT": "#aaaaaa",
    "INGREDIENT": "#cccccc",
}

# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    master = pd.read_csv(ROOT / "data" / "processed" / "master_nodes.csv")
    preds = pd.read_csv(ROOT / "outputs" / "gnn_predictions.csv")
    edges = pd.read_csv(ROOT / "data" / "labelled_data" / "edges.csv")

    df = master.merge(preds[["node_id", "gnn_pred", "gnn_prob"]], on="node_id", how="inner")
    df["predicted_label"] = df["gnn_pred"].map({1: "Critical", 0: "Non-Critical"})
    df["true_label"] = df["label"].fillna("—")

    downstream_adj, upstream_adj = build_adjacency(edges)
    return df, edges, downstream_adj, upstream_adj


@st.cache_data
def precompute_blast(_df, _edges):
    downstream_adj, upstream_adj = build_adjacency(_edges)
    dc_counts, prod_counts = {}, {}
    for nid in _df["node_id"]:
        br = blast_radius(nid, downstream_adj)
        pe = product_exposure(nid, upstream_adj, downstream_adj)
        dc_counts[nid] = br["dc_count"]
        prod_counts[nid] = pe["product_count"]
    return dc_counts, prod_counts


# ── Rule engine ──────────────────────────────────────────────────────────────

def get_suggestions(row, upstream_adj, downstream_adj, dc_count, prod_count):
    suggestions = []
    is_critical = row["gnn_pred"] == 1

    if is_critical and str(row.get("backup_level", "")).lower() == "none":
        suggestions.append(("No backup", "Add backup capacity for this node"))

    if row["node_type"] == "PLANT":
        imports = [n for n, r in upstream_adj.get(row["node_id"], []) if r == "IMPORTS_TO"]
        if len(imports) <= 1:
            suggestions.append(("Single port", "Add a secondary import path (second port)"))

    if row["node_type"] == "WAREHOUSE" and is_critical:
        replenishes = [n for n, r in downstream_adj.get(row["node_id"], []) if r == "REPLENISHES"]
        if len(replenishes) >= 3:
            suggestions.append(("High fan-out", "Add another replenishment source to reduce single-point risk"))

    recovery = row.get("recovery_days")
    if is_critical and pd.notna(recovery) and float(recovery) > 10:
        suggestions.append(("Slow recovery", f"Reduce recovery time (currently {int(recovery)} days)"))

    if is_critical and row.get("is_backup_capable") == 0:
        suggestions.append(("Not backup-capable", "Enable backup capability for this facility"))

    if is_critical and dc_count >= 5:
        suggestions.append(("Large blast radius", f"High blast radius ({dc_count} DCs) — prioritise redundancy"))

    if is_critical and prod_count >= 10:
        suggestions.append(("High product exposure", f"{prod_count} products at risk — consider alternate sourcing"))

    return suggestions


# ── Page config & data ───────────────────────────────────────────────────────

st.set_page_config(page_title="Supply-Chain Criticality", layout="wide")
st.title("Supply-Chain Criticality Dashboard")

df, edges, downstream_adj, upstream_adj = load_data()
dc_counts, prod_counts = precompute_blast(df, edges)
df["downstream_dcs"] = df["node_id"].map(dc_counts)
df["exposed_products"] = df["node_id"].map(prod_counts)

# Build prediction lookup for graph colouring
pred_lookup = dict(zip(df["node_id"], df["gnn_pred"]))

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Filters")
all_types = sorted(df["node_type"].unique())
sel_types = st.sidebar.multiselect("Node type", all_types, default=all_types)

all_regions = sorted(df["region"].dropna().unique())
sel_regions = st.sidebar.multiselect("Region", all_regions, default=all_regions)

score_range = st.sidebar.slider("Criticality score", 0.0, 1.0, (0.0, 1.0), step=0.01)

mask = (
    df["node_type"].isin(sel_types)
    & (df["region"].isin(sel_regions) | df["region"].isna())
    & df["gnn_prob"].between(*score_range)
)
filtered = df[mask].sort_values("gnn_prob", ascending=False).reset_index(drop=True)

st.sidebar.divider()
st.sidebar.header("Node selector")
node_options = filtered["node_id"].tolist()
if not node_options:
    st.warning("No nodes match the current filters.")
    st.stop()

selected = st.sidebar.selectbox("Select a node", node_options)
row = df[df["node_id"] == selected].iloc[0]
nid = row["node_id"]

# ── Pre-compute for selected node ────────────────────────────────────────────

br = blast_radius(nid, downstream_adj)
pe = product_exposure(nid, upstream_adj, downstream_adj)
suggs = get_suggestions(row, upstream_adj, downstream_adj,
                        dc_counts.get(nid, 0), prod_counts.get(nid, 0))

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_explorer, tab_graph, tab_blast, tab_errors = st.tabs(
    ["Overview", "Node Explorer", "Graph View", "Blast Radius", "Model Errors"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    critical_df = df[df["gnn_pred"] == 1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Infrastructure nodes", len(df))
    m2.metric("Predicted critical", len(critical_df))
    m3.metric("Avg criticality score", f"{df['gnn_prob'].mean():.3f}")
    m4.metric("Avg blast radius (DCs)", f"{df['downstream_dcs'].mean():.1f}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 critical nodes")
        top_crit = critical_df.nlargest(10, "gnn_prob")[
            ["node_id", "node_type", "gnn_prob", "downstream_dcs", "exposed_products"]
        ].rename(columns={
            "gnn_prob": "score", "downstream_dcs": "DCs at risk",
            "exposed_products": "products exposed",
        }).reset_index(drop=True)
        top_crit.index = top_crit.index + 1
        st.dataframe(top_crit, use_container_width=True)

    with col_right:
        st.subheader("Top 10 nodes to fix first")
        fix_df = critical_df.copy()
        max_dc = max(fix_df["downstream_dcs"].max(), 1)
        max_prod = max(fix_df["exposed_products"].max(), 1)

        rc, st_map = {}, {}
        for _, r in fix_df.iterrows():
            s = get_suggestions(r, upstream_adj, downstream_adj,
                                dc_counts.get(r["node_id"], 0), prod_counts.get(r["node_id"], 0))
            rc[r["node_id"]] = len(s)
            st_map[r["node_id"]] = "; ".join(t for _, t in s)[:80] if s else "—"

        fix_df["rule_count"] = fix_df["node_id"].map(rc)
        max_rules = max(fix_df["rule_count"].max(), 1)
        fix_df["priority"] = (
            fix_df["gnn_prob"] * 0.4
            + (fix_df["downstream_dcs"] / max_dc) * 0.3
            + (fix_df["exposed_products"] / max_prod) * 0.2
            + (fix_df["rule_count"] / max_rules) * 0.1
        )
        fix_df["suggestions"] = fix_df["node_id"].map(st_map)
        top_fix = fix_df.nlargest(10, "priority")[
            ["node_id", "node_type", "priority", "suggestions"]
        ].reset_index(drop=True)
        top_fix.index = top_fix.index + 1
        st.dataframe(
            top_fix.style.format({"priority": "{:.3f}"}, na_rep="—"),
            use_container_width=True,
        )

    st.divider()
    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.subheader("Critical nodes by type")
        type_counts = critical_df["node_type"].value_counts()
        st.bar_chart(type_counts)

    with chart_right:
        st.subheader("Critical nodes by region")
        region_counts = critical_df["region"].value_counts()
        st.bar_chart(region_counts)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: NODE EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_explorer:
    st.subheader(f"Node: {nid}")

    col_attr, col_risk = st.columns(2)

    with col_attr:
        st.markdown("**Attributes**")
        attrs = {
            "Node ID": nid,
            "Type": row["node_type"],
            "Region": row.get("region", "—"),
            "Capacity": row.get("capacity_units", "—"),
            "Throughput": row.get("throughput_units", "—"),
            "Recovery days": row.get("recovery_days", "—"),
            "Backup level": row.get("backup_level", "—"),
            "Backup capable": "Yes" if row.get("is_backup_capable") == 1 else "No",
        }
        st.table(pd.DataFrame(attrs.items(), columns=["Attribute", "Value"]))

        st.markdown("**Prediction**")
        pred_icon = "🔴" if row["gnn_pred"] == 1 else "🟢"
        true_icon = "🔴" if row.get("label") == "critical" else "🟢"
        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("Predicted", f"{pred_icon} {row['predicted_label']}")
        pc2.metric("Confidence", f"{row['gnn_prob']:.1%}")
        pc3.metric("True label", f"{true_icon} {row['true_label']}")

    with col_risk:
        st.markdown("**Why this node is risky**")
        if suggs:
            for name, _ in suggs:
                st.markdown(f"- **{name}**")
        else:
            st.success("No major risk factors detected.")

        st.markdown("**Suggested fixes**")
        if suggs:
            for name, text in suggs:
                st.markdown(f"**{name}:** {text}")
        else:
            st.success("No urgent fixes needed.")

    st.divider()
    st.markdown("**Downstream impact summary**")

    summary = f"If **{nid}** goes down"
    if br["reachable_infra"] > 0:
        summary += f", it affects **{br['reachable_infra']}** downstream infrastructure nodes"
        if br["dc_count"] > 0:
            summary += f" including **{br['dc_count']}** DCs"
    if pe["product_count"] > 0:
        summary += f" and exposes **{pe['product_count']}** products"
    summary += "."
    st.info(summary)

    dm1, dm2, dm3, dm4 = st.columns(4)
    dm1.metric("Reachable infra", br["reachable_infra"])
    dm2.metric("DCs at risk", br["dc_count"])
    dm3.metric("Products exposed", pe["product_count"])
    us_count = len(upstream_adj.get(nid, []))
    dm4.metric("Upstream connections", us_count)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: GRAPH VIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    st.subheader(f"Ego graph: {nid}")
    hops = st.radio("Expansion depth", [1, 2], horizontal=True, index=0)

    ego_nodes, ego_edges = ego_graph(nid, downstream_adj, upstream_adj, hops=hops)

    vis_nodes = []
    for n in ego_nodes:
        nt = _node_type(n)
        color = NODE_TYPE_COLORS.get(nt, "#999999")
        is_selected = n == nid
        is_critical_pred = pred_lookup.get(n) == 1

        size = 30 if is_selected else 18
        border_color = "#e74c3c" if is_critical_pred else color
        border_width = 4 if is_selected or is_critical_pred else 1

        vis_nodes.append(Node(
            id=n,
            label=n,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": color, "border": "#333333"},
            },
            size=size,
            borderWidth=border_width,
            title=f"{n} ({nt})" + (" [CRITICAL]" if is_critical_pred else ""),
        ))

    vis_edges = []
    for src, dst, rel in ego_edges:
        vis_edges.append(Edge(
            source=src,
            target=dst,
            label=rel,
            color="#888888",
        ))

    config = Config(
        width=900,
        height=500,
        directed=True,
        physics=True,
        hierarchical=True,
    )

    agraph(nodes=vis_nodes, edges=vis_edges, config=config)

    st.caption(
        "**Colors:** "
        "🔵 PORT  "
        "🟠 PLANT  "
        "🟢 WAREHOUSE  "
        "🟣 DC  "
        "⚪ PRODUCT/INGREDIENT  "
        "| **Red border** = predicted critical  "
        "| **Large node** = selected node"
    )
    st.caption(f"Showing {len(ego_nodes)} nodes and {len(ego_edges)} edges "
               f"within {hops} hop(s) of {nid}.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: BLAST RADIUS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_blast:
    st.subheader(f"Blast radius: {nid}")

    bm1, bm2, bm3, bm4 = st.columns(4)
    bm1.metric("Reachable infra nodes", br["reachable_infra"])
    bm2.metric("DCs at risk", br["dc_count"])
    bm3.metric("Products exposed", pe["product_count"])
    bm4.metric("Node type", row["node_type"])

    by_type = br["by_type"]
    if by_type:
        parts = []
        for nt in ["PLANT", "WAREHOUSE", "DC"]:
            nodes = by_type.get(nt, [])
            if nodes:
                parts.append(f"**{len(nodes)}** {nt}s")
        if parts:
            st.markdown("Downstream breakdown: " + ", ".join(parts))

    st.divider()
    ds_neighbours, us_neighbours = direct_neighbours(nid, downstream_adj, upstream_adj)

    col_ds, col_us = st.columns(2)
    with col_ds:
        st.markdown("**Direct downstream**")
        if ds_neighbours:
            st.dataframe(
                pd.DataFrame(ds_neighbours, columns=["node_id", "rel_type"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.write("None (terminal node)")
    with col_us:
        st.markdown("**Direct upstream**")
        if us_neighbours:
            st.dataframe(
                pd.DataFrame(us_neighbours, columns=["node_id", "rel_type"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.write("None (source node)")

    if br["dc_list"]:
        with st.expander(f"Show {br['dc_count']} reachable DCs"):
            st.write(", ".join(br["dc_list"]))

    if pe["products"]:
        st.divider()
        st.markdown("**Product exposure (approximate)**")
        st.caption("Products whose ingredient supply chain passes through this node. "
                   "Approximate — a product is listed if any of its required ingredients "
                   "flow through the selected node, even if alternate paths exist.")
        with st.expander(f"Show {pe['product_count']} exposed products"):
            st.write(", ".join(pe["products"]))

    st.divider()
    summary_blast = f"If **{nid}** goes down"
    if br["reachable_infra"] > 0:
        summary_blast += f", it affects **{br['reachable_infra']}** downstream infrastructure nodes"
        if br["dc_count"] > 0:
            summary_blast += f" including **{br['dc_count']}** DCs"
    if pe["product_count"] > 0:
        summary_blast += f" and exposes **{pe['product_count']}** products"
    summary_blast += "."
    st.info(summary_blast)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: MODEL ERRORS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_errors:
    st.subheader("Model error analysis")

    labelled = df[df["label"].isin(["critical", "non-critical"])].copy()
    labelled["y_true"] = (labelled["label"] == "critical").astype(int)

    fp = labelled[(labelled["gnn_pred"] == 1) & (labelled["y_true"] == 0)]
    fn = labelled[(labelled["gnn_pred"] == 0) & (labelled["y_true"] == 1)]
    total_errors = len(fp) + len(fn)

    em1, em2, em3 = st.columns(3)
    em1.metric("Total errors", total_errors)
    em2.metric("False positives", len(fp), help="Predicted critical but actually non-critical")
    em3.metric("False negatives", len(fn), help="Predicted non-critical but actually critical")

    st.divider()
    err_left, err_right = st.columns(2)

    err_cols = ["node_id", "node_type", "region", "gnn_prob", "true_label"]

    with err_left:
        st.markdown("**False positives** — predicted critical, actually non-critical")
        st.caption("Over-flagged nodes. May cause unnecessary interventions.")
        if len(fp) > 0:
            fp_display = fp[err_cols].sort_values("gnn_prob", ascending=False).reset_index(drop=True)
            fp_display.index = fp_display.index + 1
            st.dataframe(
                fp_display.rename(columns={"gnn_prob": "score"})
                .style.format({"score": "{:.3f}"}, na_rep="—"),
                use_container_width=True,
            )
        else:
            st.success("No false positives.")

    with err_right:
        st.markdown("**False negatives** — predicted non-critical, actually critical")
        st.caption("Missed critical nodes. These are the most dangerous errors.")
        if len(fn) > 0:
            fn_display = fn[err_cols].sort_values("gnn_prob", ascending=True).reset_index(drop=True)
            fn_display.index = fn_display.index + 1
            st.dataframe(
                fn_display.rename(columns={"gnn_prob": "score"})
                .style.format({"score": "{:.3f}"}, na_rep="—"),
                use_container_width=True,
            )
        else:
            st.success("No false negatives.")

    st.divider()
    st.markdown("**Errors by node type**")
    err_by_type = pd.DataFrame({
        "False positives": fp["node_type"].value_counts(),
        "False negatives": fn["node_type"].value_counts(),
    }).fillna(0).astype(int)
    if not err_by_type.empty:
        st.bar_chart(err_by_type)

    st.caption(
        "**False positive:** The model flags a node as critical when it is not. "
        "This wastes resources on unnecessary interventions. "
        "**False negative:** The model misses a truly critical node. "
        "This is the more dangerous error in supply-chain risk management."
    )
