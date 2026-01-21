# pages/4_Patient_Flow.py
import streamlit as st
import pandas as pd

from queue_mining import (
    run_queue_mining,
    plot_queue_lengths,
    plot_wait_time_distribution,
    plot_wait_time_summary,
)

from event_log_organizer import CanonicalizationConfig, StepRule

st.set_page_config(page_title="Patient Flow", layout="wide")
st.title("Patient Flow")
st.caption("Please select the state(s) you want to see — plots will update automatically.")


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
TIME_COL = mapping.get("timestamp")
STATE_COL = mapping.get("activity") or mapping.get("event") or mapping.get("state")

if not CASE_COL or not TIME_COL or not STATE_COL:
    st.error("Mapping must include: case_id, timestamp, and activity/event (state).")
    st.stop()

missing_cols = [c for c in [CASE_COL, TIME_COL, STATE_COL] if c not in df_raw.columns]
if missing_cols:
    st.error(f"Mapped columns not found in dataset: {missing_cols}")
    st.stop()

df = df_raw.copy()
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
df = df.dropna(subset=[CASE_COL, TIME_COL, STATE_COL]).copy()
df[STATE_COL] = df[STATE_COL].astype(str)

if df.empty:
    st.error("After dropping missing case/time/state rows, dataframe is empty.")
    st.stop()

unique_states = sorted(df[STATE_COL].dropna().astype(str).unique().tolist())


# -----------------------------
# Top-of-page configuration (instead of sidebar)
# -----------------------------
with st.expander("Configuration", expanded=True):
    cA, cB, cC = st.columns([1.2, 1.2, 1.0])

    with cA:
        default_display = unique_states[: min(6, len(unique_states))]
        display_states = st.multiselect(
            "States to display in plots",
            options=unique_states,
            default=default_display,
            help="Filters the plots (does not change preprocessing)."
        )

        default_terminals = [s for s in ["Discharge", "Left ED"] if s in unique_states]
        terminal_steps = st.multiselect(
            "Terminal activity values",
            options=unique_states,
            default=default_terminals,
            help="These are treated as case-ending and excluded from time-in-state summary."
        )

    with cB:
        possible_covariates = ["(none)"] + [c for c in df.columns if c not in [CASE_COL, TIME_COL, STATE_COL]]
        covariate_choice = st.selectbox(
            "Optional covariate column (for step-specific missingness rules)",
            options=possible_covariates,
            index=0,
        )
        COV_COL = None if covariate_choice == "(none)" else covariate_choice

        drop_if_midnight_steps = st.multiselect(
            "Drop steps if timestamp is midnight (00:00:00)",
            options=unique_states,
            default=[s for s in ["Ambulance Arrival"] if s in unique_states],
        )

        drop_if_cov_missing_steps = []
        if COV_COL is not None:
            drop_if_cov_missing_steps = st.multiselect(
                f"Drop steps when {COV_COL} is missing",
                options=unique_states,
                default=[],
            )

    with cC:
        sample_freq = st.selectbox(
            "Queue sampling frequency",
            options=["(none)", "5min", "15min", "30min", "1H"],
            index=2,
        )
        sample_freq = None if sample_freq == "(none)" else sample_freq

        collapse_same_state = st.checkbox(
            "Collapse consecutive same-state events",
            value=False
        )
        debug = st.checkbox("Debug logging", value=False)

    st.markdown("---")
    st.subheader("Unstable columns (missingness handling)")

    unstable_columns = st.multiselect(
        "Columns considered unstable (high missingness)",
        options=[c for c in df.columns if c not in [CASE_COL, TIME_COL, STATE_COL]],
        default=[],
    )

    unstable_mode = st.radio(
        "Missingness policy",
        options=["none", "drop_rows", "drop_cases", "drop_rows_for_steps_only"],
        index=0,
        horizontal=True,
    )

    unstable_threshold = 0.30
    if unstable_mode == "drop_cases":
        unstable_threshold = st.slider(
            "Drop case if missing fraction >",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
        )

    unstable_steps_scope = ()
    if unstable_mode == "drop_rows_for_steps_only":
        scope = st.multiselect(
            "Steps affected by unstable-row dropping",
            options=unique_states,
            default=[],
        )
        unstable_steps_scope = tuple(scope)

    st.markdown("---")
    st.subheader("Time-in-state distribution plot options")

    dist_kind = st.segmented_control(
        "Distribution plot type",
        options=["box", "violin"],
        default="box",
    )

    # Slider controls for filtering the distribution view
    # Keep these as *view filters* (do not affect preprocessing), so UI feels responsive.
    min_minutes = st.slider(
        "Minimum duration (minutes) to include in distribution",
        min_value=0.0,
        max_value=60.0,
        value=0.0,
        step=0.5,
        help="Use this to remove tiny durations created by terminal reordering/repairs."
    )

    max_minutes = st.slider(
        "Maximum duration (minutes) to include in distribution",
        min_value=10.0,
        max_value=5000.0,
        value=5000.0,
        step=10.0,
        help="Use this to cap extreme outliers so the distribution is readable."
    )


# -----------------------------
# Build preprocessing config
# -----------------------------
step_rules = {}

for step in drop_if_midnight_steps:
    step_rules[step] = StepRule(drop_if_midnight=True, priority_rank=60)

if COV_COL is not None:
    for step in drop_if_cov_missing_steps:
        step_rules[step] = StepRule(drop_if_missing_covariate=True)

for step in terminal_steps:
    if step in step_rules:
        rule = step_rules[step]
        rule.treat_as_terminal = True
        if rule.priority_rank is None:
            rule.priority_rank = 100
        step_rules[step] = rule
    else:
        step_rules[step] = StepRule(treat_as_terminal=True, priority_rank=100)

preprocessing_cfg = CanonicalizationConfig(
    case_col=CASE_COL,
    time_col=TIME_COL,
    event_col=STATE_COL,
    covariate_col=COV_COL,
    terminal_steps=tuple(terminal_steps),
    step_rules=step_rules,
    terminal_fix_mode="push_seconds",
    debug=debug,
    unstable_columns=tuple(unstable_columns),
    unstable_missing_mode=unstable_mode,
    unstable_case_missing_threshold=float(unstable_threshold),
    unstable_steps_scope=unstable_steps_scope,
)


# -----------------------------
# Run pipeline (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def _run(df_in: pd.DataFrame, preprocessing_cfg: CanonicalizationConfig,
         case_col: str, time_col: str, state_col: str, covariate_col: str | None,
         collapse: bool, sample_freq: str | None, debug: bool):
    return run_queue_mining(
        df_in,
        case_col=case_col,
        time_col=time_col,
        state_col=state_col,
        covariate_col=covariate_col,
        collapse_consecutive_same_state=collapse,
        sample_freq=sample_freq,
        parse_utc=True,
        debug=debug,
        preprocessing_cfg=preprocessing_cfg,
    )

status = st.status("Computing statistics… please be patient.", expanded=False)
try:
    status.update(label="Preprocessing event log…", state="running")
    result = _run(
        df,
        preprocessing_cfg,
        CASE_COL,
        TIME_COL,
        STATE_COL,
        COV_COL,
        collapse_same_state,
        sample_freq,
        debug,
    )
    status.update(label="Rendering charts…", state="running")
finally:
    status.update(label="Done.", state="complete")


# -----------------------------
# Filter outputs to displayed states (visual layer only)
# -----------------------------
intervals_df = result.intervals_result.intervals
if intervals_df.empty:
    st.warning("No intervals could be built after preprocessing. Try loosening drop rules.")
    st.stop()

display_set = set(display_states) if display_states else set(unique_states)

intervals_view = intervals_df.loc[intervals_df["state"].isin(display_set)].copy()
cp_view = result.queue_result.change_points.loc[result.queue_result.change_points["state"].isin(display_set)].copy()

sampled_view = None
if result.queue_result.sampled is not None:
    sampled_view = result.queue_result.sampled.loc[result.queue_result.sampled["state"].isin(display_set)].copy()

from dataclasses import replace
queue_view = replace(result.queue_result, change_points=cp_view, sampled=sampled_view)

summary_view = result.wait_summary.loc[result.wait_summary["state"].isin(display_set)].copy()

# Apply distribution *view filters* (min/max duration)
intervals_dist_view = intervals_view.copy()
intervals_dist_view = intervals_dist_view[intervals_dist_view["duration_minutes"].notna()]
intervals_dist_view = intervals_dist_view[
    (intervals_dist_view["duration_minutes"] >= float(min_minutes)) &
    (intervals_dist_view["duration_minutes"] <= float(max_minutes))
].copy()


# -----------------------------
# Plots (full-width, minimal tables)
# -----------------------------
st.divider()

st.subheader("Queue length over time")
fig_q = plot_queue_lengths(
    queue_view,
    use_sampled=True if queue_view.sampled is not None else False,
    title=f"Queue length over time (cases in {STATE_COL})"
)
st.plotly_chart(fig_q, use_container_width=True)

st.divider()

st.subheader("Time in state distribution (minutes)")
fig_dist = plot_wait_time_distribution(
    intervals_dist_view,
    kind=dist_kind,
    title=f"Time in {STATE_COL} (distribution) — {dist_kind}"
)
st.plotly_chart(fig_dist, use_container_width=True)

st.caption("Time-in-state summary (minutes)")
st.dataframe(summary_view, use_container_width=True, height=420)

st.divider()

st.subheader("Mean time in state (minutes)")
fig_mean = plot_wait_time_summary(
    summary_view,
    title=f"Mean time in {STATE_COL} (non-terminal states)"
)
st.plotly_chart(fig_mean, use_container_width=True)

st.caption(
    "Tip: Use the configuration panel at the top to choose which states to display and to set terminal/unstable-column rules."
)
