# pages/6_Anomaly_Detection.py
from __future__ import annotations

import streamlit as st
import pandas as pd

from anomaly_detection import (
    AnomalyConfig,
    detect_anomalies,
)

st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("Anomaly Detection")
st.caption("Detect anomalies: red flag, long wait, and sequence flags")

# -----------------------------
# Preconditions (lightweight)
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
ACT_COL = mapping.get("activity") or mapping.get("event") or mapping.get("state")
RESOURCE_COL = mapping.get("resource")

if not CASE_COL or not TIME_COL or not ACT_COL:
    st.error("Mapping must include: case_id, timestamp, and activity/event (state).")
    st.stop()

df = df_raw.copy()
df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce", utc=True)
df = df.dropna(subset=[CASE_COL, TIME_COL, ACT_COL]).copy()
df[ACT_COL] = df[ACT_COL].astype(str)

if RESOURCE_COL and RESOURCE_COL in df.columns:
    df[RESOURCE_COL] = df[RESOURCE_COL].astype(str)
else:
    RESOURCE_COL = None

if df.empty:
    st.error("After dropping missing case/time/activity rows, dataframe is empty.")
    st.stop()

st.session_state.setdefault("anomaly_result", None)

# ============================================================
# Config + Run button (form)
# ============================================================
with st.form("anom_form", clear_on_submit=False):
    st.subheader("Configuration")

    detectors = st.multiselect(
        "Which anomaly checks to run?",
        options=["wait_anomaly", "sequence_anomaly", "protocol_deviation"],
        default=["wait_anomaly", "sequence_anomaly"],
    )

    a1, a2 = st.columns([1.2, 1.0])
    with a1:
        cutoff_type_anom = st.selectbox(
            "Anomaly cutoff",
            options=["full_trace", "minutes_since_first_event", "n_events"],
            index=0,
        )
        cutoff_value_anom = st.slider(
            "Anomaly cutoff value",
            min_value=1,
            max_value=240,
            value=60,
            step=1,
        )
        wait_metric = st.selectbox(
            "Wait anomaly metric",
            options=["case_total_minutes", "activity_dwell_minutes_p95"],
            index=0,
        )

    with a2:
        sequence_method = st.selectbox(
            "Sequence anomaly method",
            options=["bigram_nll", "protocol_deviation"],
            index=0,
        )
        seq_q = st.slider("Sequence anomaly quantile (top % flagged)", 0.90, 0.999, 0.99, 0.001)
        wait_q = st.slider("Wait anomaly quantile (top % flagged)", 0.90, 0.999, 0.99, 0.001)

    protocol = None
    protocol_mode = "subsequence"
    if "protocol_deviation" in detectors or sequence_method == "protocol_deviation":
        protocol = st.text_input("Protocol string", value="Triage -> Registration -> Assessment")
        protocol_mode = st.selectbox(
            "Protocol matching mode",
            options=["subsequence", "prefix", "exact", "exact_with_extras"],
            index=0,
        )

    do_anoms = st.form_submit_button("Run anomaly detection", type="primary")

# ============================================================
# Heavy work ONLY runs on click
# ============================================================
if do_anoms:
    status = st.status("Running anomaly detection…", expanded=False)
    try:
        status.update(label="Preparing config…", state="running")

        # Basic toggles (depends on how detect_anomalies uses these)
        wait_metric_final = wait_metric if "wait_anomaly" in detectors else "case_total_minutes"
        seq_method_final = sequence_method if ("sequence_anomaly" in detectors or "protocol_deviation" in detectors) else "bigram_nll"
        protocol_final = protocol if ("protocol_deviation" in detectors or seq_method_final == "protocol_deviation") else None

        acfg = AnomalyConfig(
            case_col=CASE_COL,
            time_col=TIME_COL,
            act_col=ACT_COL,
            resource_col=RESOURCE_COL,
            cutoff_type=cutoff_type_anom,
            cutoff_value=int(cutoff_value_anom),
            wait_metric=wait_metric_final,
            use_quantile_threshold=True,
            quantile_threshold=float(wait_q),
            sequence_method=seq_method_final,
            seq_quantile_threshold=float(seq_q),
            protocol=protocol_final,
            protocol_mode=protocol_mode,
            top_k_activities=50,
            smoothing=1.0,
            sort_events=True,
            collapse_consecutive_duplicates=True,
        )

        status.update(label="Detecting anomalies…", state="running")
        anom_res = detect_anomalies(df, cfg=acfg, zone_intervals_df=None)
        st.session_state["anomaly_result"] = anom_res

        status.update(label="Done.", state="complete")

    except Exception as e:
        status.update(label="Failed.", state="error")
        st.exception(e)

# ============================================================
# Display
# ============================================================
anom_res = st.session_state.get("anomaly_result")
if anom_res is None:
    st.info("Configure and click **Run anomaly detection** to see results.")
else:
    st.divider()
    st.subheader("Anomalies (latest run)")

    s = anom_res.summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases", f"{s.get('n_cases', 0)}")
    c2.metric("Red flags", f"{s.get('n_red_flag', 0)}")
    c3.metric("Long-wait flags", f"{s.get('n_wait_flag', 0)}")
    c4.metric("Sequence flags", f"{s.get('n_seq_flag', 0)}")

    pc = anom_res.per_case.copy()

    st.dataframe(pc, use_container_width=True)
