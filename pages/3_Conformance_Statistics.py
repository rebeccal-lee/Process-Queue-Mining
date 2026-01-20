import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from conformance import ConformanceConfig, extract_traces, check_conformance

st.set_page_config(page_title="Conformance Statistics", layout="wide")
st.title("Conformance Statistics")

# -----------------------------
# Helpers
# -----------------------------
def collapse_consecutive(seq):
    if not seq:
        return seq
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out

def format_steps_inline(steps, max_items=12):
    """Returns 'k: a, b, c' (truncated if long)."""
    if not isinstance(steps, list):
        return "0"
    steps = [str(x) for x in steps]
    k = len(steps)
    if k == 0:
        return "0"
    show = steps[:max_items]
    suffix = " ..." if k > max_items else ""
    return f"{k}: " + ", ".join(show) + suffix

def render_trace_plot(trace, case_id, max_events=25):
    """
    Plotly path plot: nodes in order with connecting lines.
    Collapses consecutive duplicates for readability.
    """
    if not isinstance(trace, list) or len(trace) == 0:
        st.info("No trace preview available.")
        return

    trace = [str(x) for x in trace][:max_events]
    trace = collapse_consecutive(trace)

    n = len(trace)
    x = list(range(n))
    y = [0] * n

    line_x, line_y = [], []
    for i in range(n - 1):
        line_x += [x[i], x[i + 1], None]
        line_y += [0, 0, None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode="lines",
        hoverinfo="none",
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=[f"{i+1}. {t}" for i, t in enumerate(trace)],
        textposition="top center",
        hovertext=trace,
        hoverinfo="text",
        marker=dict(size=12),
    ))

    fig.update_layout(
        title=f"Trace preview — pathing for case {case_id}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=60, b=10),
        height=320,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Preconditions
# -----------------------------
df_raw = st.session_state.get("df_raw")
mapping = st.session_state.get("mapping")

if df_raw is None:
    st.info("Go to Home and upload a CSV first.")
    st.stop()

if mapping is None:
    st.info("Go to Upload Data and save a mapping first.")
    st.stop()

CASE_COL = mapping.get("case_id")
ACT_COL = mapping.get("activity")
TIME_COL = mapping.get("timestamp")

missing = [k for k, v in {"case_id": CASE_COL, "activity": ACT_COL, "timestamp": TIME_COL}.items() if not v]
if missing:
    st.error(f"Mapping missing required fields: {missing}")
    st.stop()

for c in (CASE_COL, ACT_COL, TIME_COL):
    if c not in df_raw.columns:
        st.error(f"Mapped column not found in dataset: {c}")
        st.stop()

# -----------------------------
# Prep
# -----------------------------
df = df_raw.copy()
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
df = df.dropna(subset=[CASE_COL, ACT_COL, TIME_COL])

st.session_state.setdefault("protocol_steps", [])
st.session_state.setdefault("conformance_result", None)

# -----------------------------
# Protocol builder UI
# -----------------------------
st.markdown("Build an expected **pathway** by selecting activities in order.")

activity_options = df[ACT_COL].dropna().astype(str).value_counts().index.tolist()

p1, p2 = st.columns([2, 1])

with p1:
    chosen = st.selectbox(
        "Add next step (activity)",
        options=["(select)"] + activity_options,
        index=0,
        key="proto_add_step",
    )
    if st.button("Add step", disabled=(chosen == "(select)")):
        st.session_state["protocol_steps"].append(chosen)

with p2:
    st.write("**Current pathway**")
    if st.session_state["protocol_steps"]:
        st.dataframe(
            pd.DataFrame(
                {
                    "Step #": list(range(1, len(st.session_state["protocol_steps"]) + 1)),
                    "Activity": st.session_state["protocol_steps"],
                }
            ),
            use_container_width=True,
            hide_index=True,
            height=220,
        )
    else:
        st.info("No steps yet.")

    c_back, c_clear = st.columns(2)
    with c_back:
        if st.button("⬅️ Back (remove last)", disabled=(len(st.session_state["protocol_steps"]) == 0)):
            st.session_state["protocol_steps"].pop()
    with c_clear:
        if st.button("Clear pathway", disabled=(len(st.session_state["protocol_steps"]) == 0)):
            st.session_state["protocol_steps"] = []

st.divider()

c_mode, c_run = st.columns([1, 1])
with c_mode:
    mode = st.selectbox(
        "Conformance mode",
        options=["subsequence", "exact_with_extras", "prefix", "exact"],
        index=0,
    )

run = st.button("Run conformance", type="primary", disabled=(len(st.session_state["protocol_steps"]) == 0))

# -----------------------------
# Run conformance
# -----------------------------
if run:
    cfg = ConformanceConfig(
        case_col=CASE_COL,
        act_col=ACT_COL,
        time_col=TIME_COL,
        mode=mode,  # type: ignore[arg-type]
        sort_events=True,
    )

    try:
        traces = extract_traces(
            df,
            case_col=CASE_COL,
            act_col=ACT_COL,
            time_col=TIME_COL,
            sort_events=True,
        )
        result = check_conformance(traces, st.session_state["protocol_steps"], cfg=cfg)
    except Exception as e:
        st.exception(e)
        st.stop()

    st.session_state["conformance_result"] = result

result = st.session_state.get("conformance_result")
if result is None:
    st.info("Add steps and click **Run conformance**.")
    st.stop()

per_case = result.per_case

# -----------------------------
# Headline KPIs
# -----------------------------
n_cases = len(per_case)
n_conform = int(per_case["conforms"].sum()) if n_cases else 0
pct = (n_conform / n_cases * 100.0) if n_cases else 0.0
top_dev = per_case.loc[~per_case["conforms"], "deviation_type"].value_counts().head(1)
top_dev_label = top_dev.index[0] if len(top_dev) else "none"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Cases", f"{n_cases:,}")
k2.metric("Conforming", f"{n_conform:,}")
k3.metric("Conformance rate", f"{pct:.1f}%")
k4.metric("Top deviation", str(top_dev_label))

st.subheader("Summary")
st.dataframe(result.summary, use_container_width=True)

# -----------------------------
# Drilldown
# -----------------------------
st.subheader("Drilldown")

left, right = st.columns([1, 2])

with left:
    # NEW: include conforming view
    view = st.radio("View", ["All cases", "Conforming only", "Non-conforming only"], index=2)

    dev_options = ["(any)"] + sorted(per_case["deviation_type"].astype(str).unique().tolist())
    dev_filter = st.selectbox("Deviation type", dev_options, index=0)

    case_id_options_all = sorted(per_case["case_id"].astype(str).unique().tolist())
    case_id_pick = st.selectbox("Filter to case_id (optional)", ["(any)"] + case_id_options_all, index=0)

    max_rows = st.slider("Rows to show", 0, 1000, 200)

with right:
    df_view = per_case.copy()

    if view == "Conforming only":
        df_view = df_view[df_view["conforms"] == True]
    elif view == "Non-conforming only":
        df_view = df_view[df_view["conforms"] == False]

    if dev_filter != "(any)":
        df_view = df_view[df_view["deviation_type"].astype(str) == dev_filter]

    if case_id_pick != "(any)":
        df_view = df_view[df_view["case_id"].astype(str) == case_id_pick]

    if "missing_count" in df_view.columns:
        df_view = df_view.sort_values(["conforms", "missing_count", "extra_events_count"], ascending=[False, False, False])

    show_cols = [
        "case_id", "conforms", "deviation_type",
        "missing_count", "extra_events_count",
        "trace_length", "matched_count",
    ]
    show_cols = [c for c in show_cols if c in df_view.columns]

    st.dataframe(df_view[show_cols].head(max_rows), use_container_width=True, height=520)

# -----------------------------
# Case detail (dropdown)
# -----------------------------
st.subheader("Case detail")

case_ids_available = sorted(df_view["case_id"].astype(str).unique().tolist())
if not case_ids_available:
    st.info("No cases match your drilldown filters.")
    st.stop()

selected_case = st.selectbox("Select case_id", case_ids_available, index=0)

row = per_case[per_case["case_id"].astype(str) == selected_case].iloc[0]

trace_preview = row.get("trace_preview", [])
if not isinstance(trace_preview, list):
    trace_preview = []
trace_preview = [str(x) for x in trace_preview]

missing_steps = row.get("missing_steps", [])
if not isinstance(missing_steps, list):
    missing_steps = []
missing_steps = [str(x) for x in missing_steps]

matched_positions = row.get("matched_positions", [])
if not isinstance(matched_positions, list):
    matched_positions = []

# matched steps by position
matched_steps = []
for idx in matched_positions:
    if isinstance(idx, int) and 0 <= idx < len(trace_preview):
        matched_steps.append(trace_preview[idx])

# extra events = events not used to match protocol positions (within preview window)
matched_pos_set = {i for i in matched_positions if isinstance(i, int)}
extra_events = [a for i, a in enumerate(trace_preview) if i not in matched_pos_set]

# UPDATED: include matched steps list explicitly in the table; remove bottom explicit lists
case_stats = pd.DataFrame(
    [
        ("Conforms", "Yes" if bool(row["conforms"]) else "No"),
        ("Deviation type", str(row.get("deviation_type", ""))),
        ("Trace length", int(row.get("trace_length", 0))),
        ("Protocol length", int(row.get("protocol_length", 0))),
        ("Matched steps (count)", int(row.get("matched_count", 0))),
        ("Matched steps (names)", format_steps_inline(matched_steps)),
        ("Missing steps", format_steps_inline(missing_steps)),
        ("Extra events", format_steps_inline(extra_events)),
    ],
    columns=["Metric", "Value"],
)
st.table(case_stats)

# Trace preview plot
render_trace_plot(trace_preview, selected_case, max_events=25)

st.caption(
    "Trace preview = the first 25 recorded activities for this case in time order "
    "(consecutive duplicates collapsed), to observe the path the patient took through the emergency department."
)
