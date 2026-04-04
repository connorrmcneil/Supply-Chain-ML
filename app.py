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
import plotly.graph_objects as go
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

FRIENDLY_REL = {
    "REPLENISHES": "Replenishes",
    "SHIPS_TO": "Ships to",
    "IMPORTS_TO": "Imports to",
    "SUPPLIES": "Supplies",
    "REQUIRES": "Requires",
}

# Map each data region to the states/provinces it covers for the choropleth.
# US regions use FIPS codes; Canadian provinces use ISO-3166-2 codes.
REGION_GEO = {
    "NortheastUS": {
        "label": "Northeast US",
        "states": ["CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"],
        "lat": 42.0, "lon": -74.0,
    },
    "MidwestUS": {
        "label": "Midwest US",
        "states": ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
        "lat": 42.0, "lon": -97.0,
    },
    "SoutheastUS": {
        "label": "Southeast US",
        "states": ["AL", "AR", "DE", "FL", "GA", "KY", "LA", "MD", "MS",
                   "NC", "OK", "SC", "TN", "TX", "VA", "WV", "DC"],
        "lat": 33.0, "lon": -85.0,
    },
    "Ontario": {
        "label": "Ontario",
        "provinces": ["ON"],
        "lat": 50.0, "lon": -85.0,
    },
    "Quebec": {
        "label": "Quebec",
        "provinces": ["QC"],
        "lat": 52.0, "lon": -72.0,
    },
    "BC": {
        "label": "British Columbia",
        "provinces": ["BC"],
        "lat": 54.0, "lon": -125.0,
    },
    "Prairies": {
        "label": "Prairies",
        "provinces": ["AB", "SK", "MB"],
        "lat": 53.0, "lon": -108.0,
    },
    "Atlantic": {
        "label": "Atlantic",
        "provinces": ["NB", "NS", "PE", "NL"],
        "lat": 47.0, "lon": -63.0,
    },
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
node_options = filtered["node_id"].tolist()
if not node_options:
    st.warning("No facilities match the current filters.")
    st.stop()

# Build friendly display names like "DC_022 · Ontario · Hub"
_label_map = {}
for _nid in node_options:
    _r = df[df["node_id"] == _nid].iloc[0]
    _region = str(_r.get("region", "")) if pd.notna(_r.get("region")) else ""
    _short = SHORT_TYPE_LABEL.get(_r["node_type"], _r["node_type"])
    _suffix = f"{_region} · {_short}".strip(" ·")
    _label_map[f"{_nid} — {_suffix}" if _suffix else _nid] = _nid

display_options = list(_label_map.keys())
search_query = st.sidebar.text_input(
    "Search",
    placeholder="Type to search (e.g. DC_022, Ontario, Plant...)",
    label_visibility="collapsed",
)

if search_query:
    query = search_query.strip().lower()
    matched = [d for d in display_options if query in d.lower()]
else:
    matched = display_options

if not matched:
    st.sidebar.caption(f"No results for \"{search_query}\"")
    matched = display_options

selected_display = st.sidebar.selectbox(
    "Facility",
    matched,
    label_visibility="collapsed",
)
selected = _label_map.get(selected_display, node_options[0])
row = df[df["node_id"] == selected].iloc[0]
nid = row["node_id"]
st.sidebar.caption(f"{len(matched)} of {len(display_options)} facilities shown")

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

    # ── Region map ───────────────────────────────────────────────────────────
    st.markdown("#### Facilities by region")

    region_total = df["region"].value_counts().to_dict()
    region_risk = critical_df["region"].value_counts().to_dict()

    map_rows = []
    for region_key, geo in REGION_GEO.items():
        total = region_total.get(region_key, 0)
        at_risk = region_risk.get(region_key, 0)
        map_rows.append({
            "region": geo["label"],
            "total": total,
            "at_risk": at_risk,
            "lat": geo["lat"],
            "lon": geo["lon"],
        })
    map_df = pd.DataFrame(map_rows)

    max_total = max(map_df["total"].max(), 1)

    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=map_df["lon"],
        lat=map_df["lat"],
        text=map_df.apply(
            lambda r: f"<b>{r['region']}</b><br>"
                      f"{r['total']} facilities<br>"
                      f"{r['at_risk']} high risk", axis=1),
        hoverinfo="text",
        marker=dict(
            size=map_df["total"] / max_total * 45 + 15,
            color=map_df["at_risk"],
            colorscale=[[0, "#27ae60"], [0.5, "#f39c12"], [1, "#e74c3c"]],
            opacity=0.85,
            line=dict(width=1, color="#ffffff"),
            sizemode="diameter",
        ),
        showlegend=False,
    ))

    fig.add_trace(go.Scattergeo(
        lon=map_df["lon"],
        lat=map_df["lat"],
        text=map_df["total"].astype(str),
        mode="text",
        textfont=dict(size=13, color="white", family="Arial Black"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.add_trace(go.Scattergeo(
        lon=map_df["lon"],
        lat=map_df["lat"] - 3,
        text=map_df["region"],
        mode="text",
        textfont=dict(size=9, color="#aaaaaa"),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_geos(
        scope="north america",
        showlakes=True,
        lakecolor="#0e1117",
        showocean=True,
        oceancolor="#0e1117",
        showland=True,
        landcolor="#1a2332",
        showcountries=True,
        countrycolor="#3a4a5c",
        countrywidth=1,
        showsubunits=True,
        subunitcolor="#2c3e50",
        subunitwidth=0.5,
        showcoastlines=True,
        coastlinecolor="#3a4a5c",
        coastlinewidth=0.8,
        bgcolor="#0e1117",
        projection_type="natural earth",
        center=dict(lat=45, lon=-95),
        lonaxis_range=[-135, -55],
        lataxis_range=[28, 60],
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0e1117",
        height=480,
        dragmode=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Priority table ───────────────────────────────────────────────────────
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: FACILITY DETAILS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_explorer:
    is_high_risk = row["gnn_pred"] == 1
    facility_type = FRIENDLY_TYPE.get(row["node_type"], row["node_type"])
    region_name = row.get("region", "—")

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(f"### {nid}")

    if is_high_risk:
        st.markdown(
            f"**{facility_type}** in **{region_name}** · "
            f":red[**High risk**] · "
            f"Risk score: **{row['gnn_prob']:.0%}**"
        )
        score_pct = int(row["gnn_prob"] * 100)
        if score_pct >= 75:
            score_desc = "very high likelihood of being operationally critical"
        elif score_pct >= 50:
            score_desc = "moderate-to-high likelihood of being operationally critical"
        else:
            score_desc = "moderate likelihood of being operationally critical"
        st.caption(
            f"A risk score of {score_pct}% means this facility has a {score_desc}. "
            "Disruption here would likely cascade to downstream operations."
        )
    else:
        st.markdown(
            f"**{facility_type}** in **{region_name}** · "
            f":green[**Low risk**] · "
            f"Risk score: **{row['gnn_prob']:.0%}**"
        )
        st.caption(
            f"A risk score of {int(row['gnn_prob'] * 100)}% means this facility "
            "is unlikely to cause significant disruption if taken offline."
        )

    st.markdown("")

    # ── Impact banner ────────────────────────────────────────────────────────
    impact_parts = []
    if br["reachable_infra"] > 0:
        impact_parts.append(f"**{br['reachable_infra']}** downstream facilities")
    if br["dc_count"] > 0:
        impact_parts.append(f"**{br['dc_count']}** distribution centers")
    if pe["product_count"] > 0:
        impact_parts.append(f"**{pe['product_count']}** products")

    if impact_parts:
        impact_text = (
            f"An outage at **{nid}** could disrupt "
            + ", ".join(impact_parts[:-1])
            + (f", and {impact_parts[-1]}" if len(impact_parts) > 1 else impact_parts[0])
            + ". Review the recommended actions below to reduce exposure."
        )
    else:
        impact_text = f"**{nid}** has minimal downstream exposure."
    st.info(impact_text)

    dm1, dm2, dm3 = st.columns(3)
    dm1.metric("Downstream facilities", br["reachable_infra"])
    dm2.metric("Distribution centers", br["dc_count"])
    dm3.metric("Products exposed", pe["product_count"])

    st.markdown("")

    # ── Two-column body ──────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        # ── Facility profile ─────────────────────────────────────────────────
        st.markdown("#### Facility profile")
        profile = pd.DataFrame([
            ("Type", facility_type),
            ("Region", region_name),
            ("Capacity", f"{int(row.get('capacity_units', 0)):,} units" if pd.notna(row.get("capacity_units")) else "—"),
            ("Throughput", f"{int(row.get('throughput_units', 0)):,} units" if pd.notna(row.get("throughput_units")) else "—"),
            ("Recovery time", f"{int(row.get('recovery_days', 0))} days" if pd.notna(row.get("recovery_days")) else "—"),
            ("Backup level", str(row.get("backup_level", "—")).capitalize()),
            ("Backup capable", "Yes" if row.get("is_backup_capable") == 1 else "No"),
        ], columns=["Attribute", "Value"])
        st.dataframe(profile, use_container_width=True, hide_index=True)

        # ── Upstream dependencies ────────────────────────────────────────────
        st.markdown("")
        st.markdown("#### Upstream dependencies")
        st.caption("Facilities that feed into this one. Disruption at any of these could affect this facility's operations.")
        us_neighbours = upstream_adj.get(nid, [])
        if us_neighbours:
            us_rows = []
            for uid, rel in us_neighbours:
                utype = FRIENDLY_TYPE.get(_node_type(uid), _node_type(uid))
                is_risky = pred_lookup.get(uid) == 1
                status = "High risk" if is_risky else "OK"
                us_rows.append((uid, utype, FRIENDLY_REL.get(rel, rel), status))
            us_df = pd.DataFrame(us_rows, columns=["Facility", "Type", "Relationship", "Status"])
            st.dataframe(us_df, use_container_width=True, hide_index=True)
            risky_upstream = [r[0] for r in us_rows if r[3] == "High risk"]
            if risky_upstream:
                st.caption(f":red[**Note:**] {len(risky_upstream)} upstream "
                           f"{'dependency is' if len(risky_upstream) == 1 else 'dependencies are'} "
                           "also flagged as high risk.")
        else:
            st.write("No upstream dependencies — this is a source facility.")

    with col_right:
        # ── Why flagged + actions ────────────────────────────────────────────
        if suggs:
            st.markdown("#### Why this facility is flagged")
            for name, detail in suggs:
                st.markdown(f"**{name}** — {detail}")

            st.markdown("")
            st.markdown("#### Recommended actions")
            action_map = {
                "No backup": "Establish backup capacity or identify an alternate facility.",
                "Single port": "Qualify a second import port to eliminate single-source risk.",
                "High fan-out": "Add a parallel replenishment source to distribute load.",
                "Slow recovery": f"Target recovery time under 10 days (currently {int(row.get('recovery_days', 0))}).",
                "Not backup-capable": "Invest in making this facility backup-capable.",
                "Large blast radius": "Prioritize redundancy — too many facilities depend on this one.",
                "High product exposure": "Explore alternate sourcing for the most impacted products.",
            }
            for name, _ in suggs:
                action = action_map.get(name, "Review and mitigate.")
                st.markdown(f"- {action}")

            # ── What to do first ─────────────────────────────────────────────
            st.markdown("")
            st.markdown("#### What to do first")
            if any(name == "No backup" for name, _ in suggs):
                first_step = "This facility has **no backup**. Start by identifying or provisioning backup capacity."
            elif any(name == "Slow recovery" for name, _ in suggs):
                first_step = (f"Recovery time is **{int(row.get('recovery_days', 0))} days**. "
                              "Start by reviewing what's driving the long recovery window and whether it can be shortened.")
            elif any(name == "Large blast radius" for name, _ in suggs):
                first_step = (f"This facility affects **{br['dc_count']}** distribution centers. "
                              "Start by mapping which downstream sites have no alternate source.")
            elif any(name == "Single port" for name, _ in suggs):
                first_step = "This plant depends on a single port. Start by qualifying a second import path."
            else:
                first_step = "Review the recommended actions above and prioritize based on cost and timeline."
            st.success(first_step)
        else:
            st.markdown("#### Risk assessment")
            st.success("No major risk factors identified for this facility. No immediate action required.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: NETWORK MAP
# ═══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    st.markdown(f"#### Local network: {nid}")
    st.caption("Hover over a node to see its name and type. "
               "Red borders indicate high-risk facilities.")

    hops = st.radio("Network depth", [1, 2, 3], horizontal=True, index=1,
                    help="How many steps away from the selected facility to show.")

    ego_nodes, ego_edges = ego_graph(nid, downstream_adj, upstream_adj, hops=hops)

    st.caption(f"Showing **{len(ego_nodes)}** facilities and **{len(ego_edges)}** connections "
               f"within {hops} step(s).")

    vis_nodes = []
    for n in ego_nodes:
        nt = _node_type(n)
        color = NODE_TYPE_COLORS.get(nt, "#999999")
        is_selected = n == nid
        is_critical_pred = pred_lookup.get(n) == 1

        size = 40 if is_selected else 22
        border_color = "#e74c3c" if is_critical_pred else color
        border_width = 4 if is_selected or is_critical_pred else 1

        friendly_label = FRIENDLY_TYPE.get(nt, nt)
        tooltip = f"{n}\n{friendly_label}"
        if is_critical_pred:
            tooltip += "\nHigh risk"

        # Only label the selected node; others show on hover
        label = n if is_selected else ""

        vis_nodes.append(Node(
            id=n,
            label=label,
            color={
                "background": color,
                "border": border_color,
                "highlight": {"background": color, "border": "#ffffff"},
            },
            size=size,
            borderWidth=border_width,
            title=tooltip,
            font={"color": "#ffffff", "size": 11, "face": "arial",
                  "strokeWidth": 2, "strokeColor": "#000000"},
        ))

    vis_edges = []
    for src, dst, rel in ego_edges:
        vis_edges.append(Edge(
            source=src,
            target=dst,
            color={"color": "#555555", "highlight": "#aaaaaa"},
            smooth={"type": "continuous"},
            width=1.5,
            font={"color": "#888888", "size": 9, "strokeWidth": 0, "align": "middle"},
            label=FRIENDLY_REL.get(rel, rel),
        ))

    config = Config(
        width=900,
        height=600,
        directed=True,
        physics=True,
        hierarchical=False,
        solver="forceAtlas2Based",
        stabilization=True,
        fit=True,
        minVelocity=0.75,
        maxVelocity=30,
        timestep=0.35,
    )

    graph_col, legend_col = st.columns([5, 1])

    with graph_col:
        agraph(nodes=vis_nodes, edges=vis_edges, config=config)

    with legend_col:
        st.markdown("")
        st.markdown("")
        st.markdown("**Legend**")
        legend_items = [
            ("#4a90d9", "Port"),
            ("#e8943a", "Plant"),
            ("#50b87a", "Warehouse"),
            ("#9b59b6", "Distribution Center"),
            ("#aaaaaa", "Product / Ingredient"),
        ]
        for color, label in legend_items:
            st.markdown(
                f'<span style="display:inline-block;width:12px;height:12px;'
                f'border-radius:50%;background:{color};margin-right:8px;'
                f'vertical-align:middle;"></span>'
                f'<span style="color:#cccccc;font-size:13px;">{label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown("")
        st.markdown(
            '<span style="display:inline-block;width:12px;height:12px;'
            'border-radius:50%;background:#333;border:2px solid #e74c3c;'
            'margin-right:8px;vertical-align:middle;"></span>'
            '<span style="color:#cccccc;font-size:13px;">High risk</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span style="display:inline-block;width:16px;height:16px;'
            'border-radius:50%;background:#555;margin-right:6px;'
            'vertical-align:middle;"></span>'
            '<span style="color:#cccccc;font-size:13px;">Selected</span>',
            unsafe_allow_html=True,
        )

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
