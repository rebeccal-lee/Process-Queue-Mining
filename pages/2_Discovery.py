# pages/2_Discovery.py
import streamlit as st
import pandas as pd

from discovery import DFGViewConfig, build_dfg_view

st.set_page_config(page_title="Discovery", layout="wide")
st.title("Discovery")

# 1) Get data uploaded on Home
df_raw = st.session_state.get("df_raw")
if df_raw is None:
    st.info("Go to Home and upload a CSV first.")
    st.stop()

# 2) ---- EDIT THESE to match your dataset ----
CASE_COL = "visit_id"
ACT_COL  = "event"
TIME_COL = "timestamp"

missing = [c for c in (CASE_COL, ACT_COL, TIME_COL) if c not in df_raw.columns]
if missing:
    st.error(
        "Static Discovery config points to columns that don't exist.\n\n"
        f"Missing: {missing}\n\n"
        f"Available columns: {list(df_raw.columns)}"
    )
    st.stop()

# 3) Minimal prep: parse timestamp + drop bad rows
event_log_df = df_raw.copy()
event_log_df[TIME_COL] = pd.to_datetime(event_log_df[TIME_COL], errors="coerce")
event_log_df = event_log_df.dropna(subset=[CASE_COL, ACT_COL, TIME_COL])

# -----------------------------
# Sidebar: Simple unified filter
# -----------------------------
with st.sidebar:
    st.subheader("Filter (optional)")

    all_cols = list(event_log_df.columns)

    filter_col = st.selectbox(
        "Filter by column",
        ["(none)"] + all_cols,
        index=0,
        key="filter_col",
    )

    filter_kind = None
    num_range = None
    cat_keep = None

    if filter_col != "(none)":
        s = event_log_df[filter_col]
        s_num = pd.to_numeric(s, errors="coerce")
        numeric_ratio = float(s_num.notna().mean())

        if numeric_ratio > 0.8:
            filter_kind = "numeric"
            lo = float(s_num.min())
            hi = float(s_num.max())

            if lo == hi:
                st.info(f"{filter_col} has a single numeric value: {lo}")
            else:
                num_range = st.slider(
                    f"{filter_col} range",
                    min_value=lo,
                    max_value=hi,
                    value=(lo, hi),
                    key="num_range",
                )
        else:
            filter_kind = "categorical"
            opts = sorted(s.dropna().astype(str).unique().tolist())[:5000]
            if not opts:
                st.warning(f"No non-missing values found in {filter_col}.")
            else:
                cat_keep = st.multiselect(
                    f"{filter_col} values",
                    options=opts,
                    default=opts,   # keep all by default
                    key="cat_keep",
                )

    st.divider()
    st.subheader("Graph settings")

    min_edge_count = st.slider("Min edge count", 1, 50, 1, key="min_edge_count")
    top_k_edges = st.selectbox("Top-K edges", [None, 25, 50, 100], index=2, key="top_k_edges")
    max_nodes = st.selectbox("Max nodes", [None, 25, 40, 60, 100], index=2, key="max_nodes")
    rankdir = st.selectbox("Layout", ["LR", "TB"], index=0, key="rankdir")
    drop_self_loops = st.checkbox("Drop self loops", value=False, key="drop_self_loops")
    show_edge_labels = st.checkbox("Show edge labels", value=True, key="show_edge_labels")
    node_label_with_count = st.checkbox("Show counts in node labels", value=True, key="node_label_with_count")

# -----------------------------
# Apply filter (if any)
# -----------------------------
filtered = event_log_df

if filter_col != "(none)":
    if filter_kind == "numeric" and num_range is not None:
        s_num = pd.to_numeric(filtered[filter_col], errors="coerce")
        filtered = filtered[s_num.between(num_range[0], num_range[1], inclusive="both")]
    elif filter_kind == "categorical" and cat_keep is not None:
        if len(cat_keep) == 0:
            filtered = filtered.iloc[0:0]
        else:
            filtered = filtered[filtered[filter_col].astype(str).isin(set(cat_keep))]

st.caption(f"Rows after filter: {len(filtered):,} (of {len(event_log_df):,})")

if len(filtered) == 0:
    st.warning("No rows match your filter. Adjust the filter to see results.")
    st.stop()

# -----------------------------
# Build DFG + plot
# -----------------------------
cfg = DFGViewConfig(
    min_edge_count=min_edge_count,
    top_k_edges=top_k_edges,
    view_mode="full",
    rankdir=rankdir,
    show_edge_labels=show_edge_labels,
    node_label_with_count=node_label_with_count,
    max_nodes=max_nodes,
)

try:
    dfg_out, fig = build_dfg_view(
        filtered,
        cfg=cfg,
        case_col=CASE_COL,
        act_col=ACT_COL,
        time_col=TIME_COL,
        drop_self_loops=drop_self_loops,
    )
except Exception as e:
    st.exception(e)
    st.stop()

# -----------------------------
# Page layout: tables by default
# -----------------------------
st.subheader("Directly-Follows Graph")
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("DFG Tables")
left, right = st.columns(2)
with left:
    st.markdown("### Nodes")
    st.dataframe(dfg_out.nodes, use_container_width=True, height=420)

with right:
    st.markdown("### Edges")
    st.dataframe(dfg_out.edges, use_container_width=True, height=420)




