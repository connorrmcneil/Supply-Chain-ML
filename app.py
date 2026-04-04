"""
Supply Chain Risk Dashboard
============================
Multi-tab Streamlit app showing predicted risk, network views,
downstream impact analysis, and model accuracy review.

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

FRIENDLY_TYPE = {
    "PORT": "Port",
    "PLANT": "Plant",
    "WAREHOUSE": "Warehouse",
    "DC": "Distribution Center",
    "PRODUCT": "Product",
    "INGREDIENT": "Ingredient",
}

SHORT_TYPE_LABEL = {
    "PORT": "Port",
    "PLANT": "Plant",
    "WAREHOUSE": "Warehouse",
    "DC": "Hub",
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
        suggestions.append(("No backup", "Add backup capacity for this facility"))

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
        suggestions.append(("Large blast radius", f"High downstream impact ({dc_count} distribution centers affected) — prioritize redundancy"))

    if is_critical and prod_count >= 10:
        suggestions.append(("High product exposure", f"{prod_count} products at risk — consider alternate sourcing"))

    return suggestions


# ── Page config & data ───────────────────────────────────────────────────────

st.set_page_config(page_title="Supply Chain Risk", layout="wide")

# ── Sidebar & pill styling ───────────────────────────────────────────────────

st.markdown("""
<style>
    /* Uniform-width pills in sidebar multiselects */
    section[data-testid="stSidebar"] [data-baseweb="tag"] {
        flex: 0 0 100% !important;
        max-width: 100% !important;
    }
    /* Sidebar section spacing */
    section[data-testid="stSidebar"] hr {
        margin: 0.6rem 0;
    }
    /* Profile button styling */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a8a7d, #17a08e);
        border: none;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
</style>
""", unsafe_allow_html=True)

df, edges, downstream_adj, upstream_adj = load_data()
dc_counts, prod_counts = precompute_blast(df, edges)
df["downstream_dcs"] = df["node_id"].map(dc_counts)
df["exposed_products"] = df["node_id"].map(prod_counts)

pred_lookup = dict(zip(df["node_id"], df["gnn_pred"]))

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("## Supply Chain Risk Dashboard")
st.caption("Identify high-risk facilities, understand downstream impact, and prioritize actions.")

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.markdown("**FILTERS**")

st.sidebar.caption("Facility Types")
all_types = sorted(df["node_type"].unique())
friendly_options = [FRIENDLY_TYPE.get(t, t) for t in all_types]
reverse_type = {v: k for k, v in FRIENDLY_TYPE.items()}
sel_friendly = st.sidebar.multiselect(
    "Facility Types", friendly_options, default=friendly_options,
    label_visibility="collapsed",
)
sel_types = [reverse_type.get(f, f) for f in sel_friendly]

st.sidebar.caption("Regions")
all_regions = sorted(df["region"].dropna().unique())
sel_regions = st.sidebar.multiselect(
    "Regions", all_regions, default=all_regions,
    label_visibility="collapsed",
)

st.sidebar.divider()

st.sidebar.markdown("**ANALYSIS**")

score_range = st.sidebar.slider("Risk Score", 0.0, 1.0, (0.0, 1.0), step=0.01)

mask = (
    df["node_type"].isin(sel_types)
    & (df["region"].isin(sel_regions) | df["region"].isna())
    & df["gnn_prob"].between(*score_range)
)
filtered = df[mask].sort_values("gnn_prob", ascending=False).reset_index(drop=True)

st.sidebar.divider()

st.sidebar.markdown("**SELECT FACILITY**")
st.sidebar.caption("Facility ID")
node_options = filtered["node_id"].tolist()
if not node_options:
    st.warning("No facilities match the current filters.")
    st.stop()

# Build friendly display names like "DC_022 (Ontario Hub)"
_label_map = {}
for _nid in node_options:
    _r = df[df["node_id"] == _nid].iloc[0]
    _region = str(_r.get("region", "")) if pd.notna(_r.get("region")) else ""
    _short = SHORT_TYPE_LABEL.get(_r["node_type"], "")
    _suffix = f"{_region} {_short}".strip()
    _label_map[f"{_nid} ({_suffix})" if _suffix else _nid] = _nid

display_options = list(_label_map.keys())
selected_display = st.sidebar.selectbox(
    "Search facility",
    display_options,
    label_visibility="collapsed",
)
selected = _label_map.get(selected_display, node_options[0])
row = df[df["node_id"] == selected].iloc[0]
nid = row["node_id"]

# ── Pre-compute for selected facility ────────────────────────────────────────

br = blast_radius(nid, downstream_adj)
pe = product_exposure(nid, upstream_adj, downstream_adj)
suggs = get_suggestions(row, upstream_adj, downstream_adj,
                        dc_counts.get(nid, 0), prod_counts.get(nid, 0))

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_overview, tab_explorer, tab_graph, tab_blast, tab_errors = st.tabs(
    ["Overview", "Facility Details", "Network Map", "Downstream Impact", "Model Accuracy"]
)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    critical_df = df[df["gnn_pred"] == 1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total facilities", len(df))
    m2.metric("Flagged as high risk", len(critical_df))
    m3.metric("Avg risk score", f"{df['gnn_prob'].mean():.2f}")
    m4.metric("Avg downstream DCs affected", f"{df['downstream_dcs'].mean():.1f}")

    st.markdown("")

    st.markdown("#### Highest-priority facilities to address")
    st.caption("Ranked by a combination of risk score, downstream impact, product exposure, and number of actionable findings.")

    fix_df = critical_df.copy()
    max_dc = max(fix_df["downstream_dcs"].max(), 1)
    max_prod = max(fix_df["exposed_products"].max(), 1)

    rc, st_map = {}, {}
    for _, r in fix_df.iterrows():
        s = get_suggestions(r, upstream_adj, downstream_adj,
                            dc_counts.get(r["node_id"], 0), prod_counts.get(r["node_id"], 0))
        rc[r["node_id"]] = len(s)
        st_map[r["node_id"]] = "; ".join(t for _, t in s)[:120] if s else "No specific findings"

    fix_df["rule_count"] = fix_df["node_id"].map(rc)
    max_rules = max(fix_df["rule_count"].max(), 1)
    fix_df["priority"] = (
        fix_df["gnn_prob"] * 0.4
        + (fix_df["downstream_dcs"] / max_dc) * 0.3
        + (fix_df["exposed_products"] / max_prod) * 0.2
        + (fix_df["rule_count"] / max_rules) * 0.1
    )
    fix_df["suggested_actions"] = fix_df["node_id"].map(st_map)
    top_fix = fix_df.nlargest(10, "priority")[
        ["node_id", "node_type", "gnn_prob", "downstream_dcs", "exposed_products", "suggested_actions"]
    ].rename(columns={
        "node_id": "Facility",
        "node_type": "Type",
        "gnn_prob": "Risk score",
        "downstream_dcs": "DCs affected",
        "exposed_products": "Products at risk",
        "suggested_actions": "Recommended actions",
    }).reset_index(drop=True)
    top_fix.index = top_fix.index + 1
    st.dataframe(
        top_fix.style.format({"Risk score": "{:.2f}"}, na_rep="—"),
        use_container_width=True,
    )

    st.markdown("")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### High-risk facilities by type")
        type_counts = critical_df["node_type"].value_counts().rename(
            index=FRIENDLY_TYPE
        )
        st.bar_chart(type_counts)

    with chart_right:
        st.markdown("#### High-risk facilities by region")
        region_counts = critical_df["region"].value_counts()
        st.bar_chart(region_counts)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: FACILITY DETAILS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_explorer:
    is_high_risk = row["gnn_pred"] == 1
    risk_badge = "High risk" if is_high_risk else "Low risk"
    risk_color = "red" if is_high_risk else "green"

    st.markdown(f"### {nid}")
    st.markdown(
        f"**{FRIENDLY_TYPE.get(row['node_type'], row['node_type'])}** · "
        f"{row.get('region', '—')} · "
        f":{risk_color}[{risk_badge}] · "
        f"Risk score: **{row['gnn_prob']:.0%}**"
    )

    st.markdown("")

    col_details, col_actions = st.columns(2)

    with col_details:
        st.markdown("#### Facility profile")
        details = {
            "Capacity": f"{int(row.get('capacity_units', 0)):,}" if pd.notna(row.get("capacity_units")) else "—",
            "Throughput": f"{int(row.get('throughput_units', 0)):,}" if pd.notna(row.get("throughput_units")) else "—",
            "Recovery time": f"{int(row.get('recovery_days', 0))} days" if pd.notna(row.get("recovery_days")) else "—",
            "Backup level": str(row.get("backup_level", "—")).capitalize(),
            "Backup capable": "Yes" if row.get("is_backup_capable") == 1 else "No",
        }
        st.table(pd.DataFrame(details.items(), columns=["", "Value"]))

    with col_actions:
        if suggs:
            st.markdown("#### Why this facility is flagged")
            for name, _ in suggs:
                st.markdown(f"- {name}")

            st.markdown("")
            st.markdown("#### Recommended actions")
            for _, text in suggs:
                st.markdown(f"- {text}")
        else:
            st.markdown("#### Risk assessment")
            st.success("No major risk factors identified for this facility.")

    st.markdown("")

    summary = f"If **{nid}** goes offline"
    if br["reachable_infra"] > 0:
        summary += f", **{br['reachable_infra']}** downstream facilities are affected"
        if br["dc_count"] > 0:
            summary += f" (including **{br['dc_count']}** distribution centers)"
    if pe["product_count"] > 0:
        summary += f" and **{pe['product_count']}** products are exposed to disruption"
    summary += "."
    st.info(summary)

    dm1, dm2, dm3 = st.columns(3)
    dm1.metric("Downstream facilities at risk", br["reachable_infra"])
    dm2.metric("Distribution centers at risk", br["dc_count"])
    dm3.metric("Products exposed", pe["product_count"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: NETWORK MAP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    st.markdown(f"#### Local network: {nid}")
    st.caption("Shows the facilities directly connected to the selected facility in the supply chain.")

    hops = st.radio("Network depth", [1, 2], horizontal=True, index=0,
                    help="How many steps away from the selected facility to show.")

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

        friendly_label = FRIENDLY_TYPE.get(nt, nt)
        tooltip = f"{n} ({friendly_label})"
        if is_critical_pred:
            tooltip += " — High risk"

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
            title=tooltip,
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
        "**Legend:** "
        "Blue = Port · "
        "Orange = Plant · "
        "Green = Warehouse · "
        "Purple = Distribution Center · "
        "Gray = Product/Ingredient · "
        "Red border = High risk · "
        "Large = Selected facility"
    )
    st.caption(f"{len(ego_nodes)} facilities and {len(ego_edges)} connections shown "
               f"within {hops} step(s) of {nid}.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: DOWNSTREAM IMPACT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_blast:
    st.markdown(f"#### Downstream impact: {nid}")
    st.caption("What happens downstream in the supply chain if this facility goes offline.")

    bm1, bm2, bm3 = st.columns(3)
    bm1.metric("Downstream facilities affected", br["reachable_infra"])
    bm2.metric("Distribution centers at risk", br["dc_count"])
    bm3.metric("Products exposed", pe["product_count"])

    by_type = br["by_type"]
    if by_type:
        parts = []
        for nt, label in [("PLANT", "Plants"), ("WAREHOUSE", "Warehouses"), ("DC", "Distribution Centers")]:
            nodes = by_type.get(nt, [])
            if nodes:
                parts.append(f"**{len(nodes)}** {label}")
        if parts:
            st.markdown("Breakdown: " + " · ".join(parts))

    st.markdown("")

    summary_blast = f"If **{nid}** goes offline"
    if br["reachable_infra"] > 0:
        summary_blast += f", **{br['reachable_infra']}** downstream facilities are affected"
        if br["dc_count"] > 0:
            summary_blast += f" (including **{br['dc_count']}** distribution centers)"
    if pe["product_count"] > 0:
        summary_blast += f" and **{pe['product_count']}** products could be disrupted"
    summary_blast += "."
    st.info(summary_blast)

    st.markdown("")
    ds_neighbours, us_neighbours = direct_neighbours(nid, downstream_adj, upstream_adj)

    col_ds, col_us = st.columns(2)
    with col_ds:
        st.markdown("**Direct downstream connections**")
        if ds_neighbours:
            ds_data = pd.DataFrame(ds_neighbours, columns=["Facility", "Relationship"])
            st.dataframe(ds_data, use_container_width=True, hide_index=True)
        else:
            st.write("None — this is a terminal facility.")
    with col_us:
        st.markdown("**Direct upstream connections**")
        if us_neighbours:
            us_data = pd.DataFrame(us_neighbours, columns=["Facility", "Relationship"])
            st.dataframe(us_data, use_container_width=True, hide_index=True)
        else:
            st.write("None — this is a source facility.")

    if br["dc_list"]:
        with st.expander(f"View {br['dc_count']} affected distribution centers"):
            dc_df = pd.DataFrame({"Distribution Center": br["dc_list"]})
            st.dataframe(dc_df, use_container_width=True, hide_index=True)

    if pe["products"]:
        st.markdown("")
        with st.expander(f"View {pe['product_count']} exposed products"):
            st.caption(
                "Products whose ingredient supply chain passes through this facility. "
                "A product is listed if any of its required ingredients flow through here, "
                "even if alternate supply paths exist."
            )
            prod_df = pd.DataFrame({"Product": pe["products"]})
            st.dataframe(prod_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: MODEL ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════

with tab_errors:
    st.markdown("#### Model accuracy review")
    st.caption(
        "Compares model predictions against known labels to surface potential blind spots. "
        "Use this to understand where the model may over- or under-flag risk."
    )

    labelled = df[df["label"].isin(["critical", "non-critical"])].copy()
    labelled["y_true"] = (labelled["label"] == "critical").astype(int)

    fp = labelled[(labelled["gnn_pred"] == 1) & (labelled["y_true"] == 0)]
    fn = labelled[(labelled["gnn_pred"] == 0) & (labelled["y_true"] == 1)]
    total_errors = len(fp) + len(fn)

    em1, em2, em3 = st.columns(3)
    em1.metric("Total mismatches", total_errors)
    em2.metric(
        "Over-flagged",
        len(fp),
        help="Facilities flagged as high risk that are actually low risk. May lead to unnecessary interventions.",
    )
    em3.metric(
        "Missed risks",
        len(fn),
        help="Facilities that are actually high risk but were not flagged. These are the most important to review.",
    )

    st.markdown("")

    err_cols = ["node_id", "node_type", "region", "gnn_prob"]

    err_left, err_right = st.columns(2)

    with err_left:
        st.markdown("**Over-flagged facilities**")
        st.caption("Flagged as high risk but actually low risk.")
        if len(fp) > 0:
            fp_display = fp[err_cols].sort_values("gnn_prob", ascending=False).reset_index(drop=True)
            fp_display.index = fp_display.index + 1
            st.dataframe(
                fp_display.rename(columns={
                    "node_id": "Facility", "node_type": "Type",
                    "region": "Region", "gnn_prob": "Risk score",
                }).style.format({"Risk score": "{:.2f}"}, na_rep="—"),
                use_container_width=True,
            )
        else:
            st.success("No over-flagged facilities.")

    with err_right:
        st.markdown("**Missed high-risk facilities**")
        st.caption("Actually high risk but not flagged by the model. Review these carefully.")
        if len(fn) > 0:
            fn_display = fn[err_cols].sort_values("gnn_prob", ascending=True).reset_index(drop=True)
            fn_display.index = fn_display.index + 1
            st.dataframe(
                fn_display.rename(columns={
                    "node_id": "Facility", "node_type": "Type",
                    "region": "Region", "gnn_prob": "Risk score",
                }).style.format({"Risk score": "{:.2f}"}, na_rep="—"),
                use_container_width=True,
            )
        else:
            st.success("No missed high-risk facilities.")

    with st.expander("Mismatches by facility type"):
        err_by_type = pd.DataFrame({
            "Over-flagged": fp["node_type"].value_counts(),
            "Missed risks": fn["node_type"].value_counts(),
        }).fillna(0).astype(int)
        if not err_by_type.empty:
            err_by_type.index = err_by_type.index.map(lambda x: FRIENDLY_TYPE.get(x, x))
            st.bar_chart(err_by_type)
        else:
            st.write("No mismatches to display.")
