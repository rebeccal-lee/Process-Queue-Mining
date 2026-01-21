# pages/5_Prediction_Anomaly.py
import streamlit as st
import pandas as pd
import numpy as np

from predictive_analytics import (
    PredictionConfig,
    TargetBuildConfig,
    build_required_target,
    build_prediction_dataset,
    train_model,
    predict,
    evaluate_model,
    summarize_target_with_ci,
    DEFAULT_METRIC_SETS,
    bootstrap_prediction_ci,
)

st.set_page_config(page_title="Patient Prediction", layout="wide")
st.title("Patient Prediction")
st.caption("Predict: P(admission), P(LWBS), and remaining time-to-PIA (regression).")

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
ACT_COL = mapping.get("activity") or mapping.get("event") or mapping.get("state")
RESOURCE_COL = mapping.get("resource")

if not CASE_COL or not TIME_COL or not ACT_COL:
    st.error("Mapping must include: case_id, timestamp, and activity/event (state).")
    st.stop()

missing_cols = [c for c in [CASE_COL, TIME_COL, ACT_COL] if c not in df_raw.columns]
if missing_cols:
    st.error(f"Mapped columns not found in dataset: {missing_cols}")
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

unique_acts = sorted(df[ACT_COL].dropna().unique().tolist())

# -----------------------------
# UI Helpers
# -----------------------------
def safe_float(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def fmt_num(x: float, fmt: str) -> str:
    """
    fmt should be like ".3f" or ".1f"
    """
    x = safe_float(x)
    if not np.isfinite(x):
        return "N/A"
    return format(x, fmt)

def fmt_ci(val: float, lo: float, hi: float, fmt: str) -> str:
    """
    Always returns a Streamlit-metric-friendly string:
      0.123 (0.100–0.200)
    """
    val = safe_float(val)
    lo = safe_float(lo)
    hi = safe_float(hi)
    if not (np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi)):
        return "N/A"
    return f"{format(val, fmt)} ({format(lo, fmt)}–{format(hi, fmt)})"

# -----------------------------
# Configuration (hidden)
# -----------------------------
# Provide friendly names but map to DEFAULT_METRIC_SETS keys
metric_key_binary_default = "Binary" if "Binary" in DEFAULT_METRIC_SETS else list(DEFAULT_METRIC_SETS.keys())[0]
metric_key_time_default = "Regression" if "Regression" in DEFAULT_METRIC_SETS else list(DEFAULT_METRIC_SETS.keys())[0]

# persistent defaults
if "metric_key_binary" not in st.session_state:
    st.session_state["metric_key_binary"] = metric_key_binary_default
if "metric_key_time" not in st.session_state:
    st.session_state["metric_key_time"] = metric_key_time_default

with st.expander("Configuration (click to expand)", expanded=False):
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.2])

    with c1:
        pia_activity = st.selectbox(
            "Define PIA activity (first provider assessment event)",
            options=unique_acts,
            index=0 if len(unique_acts) else 0,
            help="Pick the activity that represents 'Physician Initial Assessment' (often 'Assessment').",
        )
        cutoff_minutes = st.slider(
            "Prediction cutoff window (minutes since arrival)",
            min_value=0,
            max_value=240,
            value=int(st.session_state.get("cutoff_minutes", 30)),
            step=5,
            help="Features are built from events up to this cutoff time.",
        )
        st.session_state["cutoff_minutes"] = int(cutoff_minutes)

    with c2:
        candidate_static = [c for c in df.columns if c not in [CASE_COL, TIME_COL, ACT_COL]]
        default_static = [c for c in ["age", "triage_code", "triage_desc", "initial_zone", "gender"] if c in candidate_static]

        static_feature_cols = st.multiselect(
            "Profile fields (static features)",
            options=candidate_static,
            default=st.session_state.get("static_feature_cols", default_static),
            help="Constant per visit and become user inputs.",
        )
        st.session_state["static_feature_cols"] = static_feature_cols

        disposition_col_guess = "disposition_desc" if "disposition_desc" in df.columns else None
        disposition_col = st.selectbox(
            "Disposition column (for admission/LWBS targets)",
            options=["(none)"] + candidate_static,
            index=(
                1 + candidate_static.index(disposition_col_guess)
                if disposition_col_guess in candidate_static
                else 0
            ),
            help="Needed to derive admission & LWBS from disposition text/codes.",
        )
        st.session_state["disposition_col"] = disposition_col

    with c3:
        st.markdown("**Models**")
        model_type_bin = st.selectbox("Binary model (Admission/LWBS)", options=["logreg", "rf"], index=0)
        model_type_reg = st.selectbox("Regression model (Time-to-PIA)", options=["rf", "logreg"], index=0)

    with c4:
        st.markdown("**Uncertainty & metrics**")
        show_ci = st.checkbox("Show 95% CIs", value=True)
        show_patient_ci = st.checkbox("Patient-level CI (bootstrap refit, slower)", value=False)
        patient_ci_boot = st.slider("Bootstrap reps (patient CI)", 50, 300, 150, 25) if show_patient_ci else 0

        # Metric selectors moved here
        # Rename display labels but store DEFAULT_METRIC_SETS key
        all_keys = list(DEFAULT_METRIC_SETS.keys())

        metric_key_binary = st.selectbox(
            "Metric set: Binary (yes/no)",
            options=all_keys,
            index=all_keys.index(st.session_state["metric_key_binary"]) if st.session_state["metric_key_binary"] in all_keys else 0,
        )
        metric_key_time = st.selectbox(
            "Metric set: Time (minutes)",
            options=all_keys,
            index=all_keys.index(st.session_state["metric_key_time"]) if st.session_state["metric_key_time"] in all_keys else 0,
        )
        st.session_state["metric_key_binary"] = metric_key_binary
        st.session_state["metric_key_time"] = metric_key_time

# fallbacks on rerun
pia_activity = st.session_state.get("pia_activity", unique_acts[0] if unique_acts else "PIA")
st.session_state["pia_activity"] = pia_activity
cutoff_minutes = int(st.session_state.get("cutoff_minutes", 30))
static_feature_cols = st.session_state.get("static_feature_cols", [])
disposition_col = st.session_state.get("disposition_col", "(none)")
show_ci = locals().get("show_ci", True)
show_patient_ci = locals().get("show_patient_ci", False)
patient_ci_boot = int(locals().get("patient_ci_boot", 0))
model_type_bin = locals().get("model_type_bin", st.session_state.get("model_type_bin", "logreg"))
model_type_reg = locals().get("model_type_reg", st.session_state.get("model_type_reg", "rf"))
st.session_state["model_type_bin"] = model_type_bin
st.session_state["model_type_reg"] = model_type_reg

metric_key_binary = st.session_state.get("metric_key_binary")
metric_key_time = st.session_state.get("metric_key_time")
metric_keys_binary = DEFAULT_METRIC_SETS.get(metric_key_binary, [])
metric_keys_time = DEFAULT_METRIC_SETS.get(metric_key_time, [])

st.divider()

# ============================================================
# Patient profile input
# ============================================================
st.subheader("Patient profile")
st.caption("Fill the fields below, then click **Run predictions**.")

profile_cols = static_feature_cols[:] if static_feature_cols else []

p1, p2 = st.columns([1.1, 0.9])

with p1:
    arrival_dt = st.date_input("Arrival date", value=pd.Timestamp.utcnow().date())
    arrival_tm = st.time_input("Arrival time (local)", value=pd.Timestamp.now().time())
    arrival_ts = pd.Timestamp.combine(arrival_dt, arrival_tm)

    n_events_so_far = st.slider("Events observed so far", min_value=0, max_value=15, value=0, step=1)
    elapsed_so_far = st.slider(
        "Minutes since arrival (so far)",
        min_value=0,
        max_value=240,
        value=min(30, cutoff_minutes),
        step=5,
    )

    last_activity = st.selectbox("Last observed activity (optional)", options=["(none)"] + (unique_acts[:50]), index=0)

with p2:
    inputs = {}
    for col in profile_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            s_clean = pd.to_numeric(s, errors="coerce")
            lo = float(np.nanpercentile(s_clean, 1)) if s_clean.notna().any() else 0.0
            hi = float(np.nanpercentile(s_clean, 99)) if s_clean.notna().any() else 100.0
            default = float(np.nanmedian(s_clean)) if s_clean.notna().any() else 0.0
            inputs[col] = st.number_input(col, value=default, min_value=lo, max_value=hi)
        else:
            opts = s.astype(str).dropna().value_counts().head(50).index.tolist()
            if len(opts) == 0:
                inputs[col] = st.text_input(col, value="")
            else:
                inputs[col] = st.selectbox(col, options=opts, index=0)

# ============================================================
# Single button: build if needed + score
# ============================================================
run = st.button("Run predictions (build models if needed)", type="primary")

def current_signature() -> tuple:
    # only config that *affects training/features* goes here
    return (
        pia_activity,
        cutoff_minutes,
        tuple(static_feature_cols) if static_feature_cols else tuple(),
        disposition_col,
        model_type_bin,
        model_type_reg,
        CASE_COL,
        TIME_COL,
        ACT_COL,
        RESOURCE_COL,
    )

sig = current_signature()
needs_rebuild = (st.session_state.get("model_signature") != sig) or ("X_all" not in st.session_state)

if not run:
    st.stop()

# ============================================================
# Build models (if needed)
# ============================================================
if needs_rebuild:
    status = st.status("Building models…", expanded=False)
    try:
        status.update(label="Preparing configs…", state="running")

        pred_cfg = PredictionConfig(
            case_col=CASE_COL,
            time_col=TIME_COL,
            act_col=ACT_COL,
            resource_col=RESOURCE_COL,
            static_feature_cols=static_feature_cols if static_feature_cols else None,
            cutoff_type="minutes_since_first_event",
            cutoff_value=int(cutoff_minutes),
            top_k_activities=50,
            include_time_features=True,
            include_activity_counts=True,
            include_last_activity=True,
            include_resource_counts=False,
            sort_events=True,
            dropna_timestamps=True,
        )

        tcfg = TargetBuildConfig(
            case_col=CASE_COL,
            time_col=TIME_COL,
            act_col=ACT_COL,
            disposition_col=None if disposition_col == "(none)" else disposition_col,
            pia_activity=pia_activity,
            require_pia=True,
            cap_remaining_minutes=24 * 60.0,
        )

        status.update(label="Building feature matrix (X)…", state="running")
        dataset_X_only = build_prediction_dataset(df, target_df=None, target_col=None, cfg=pred_cfg)
        X_all = dataset_X_only.X

        admission_dataset = lwbs_dataset = None
        t_adm = t_lwbs = None
        spec_adm = spec_lwbs = None

        if tcfg.disposition_col is not None:
            status.update(label="Building admission dataset…", state="running")
            t_adm, spec_adm = build_required_target(df, which="prob_admission", tcfg=tcfg, pred_cfg=pred_cfg)
            admission_dataset = build_prediction_dataset(df, target_df=t_adm, target_case_col=CASE_COL, target_col=spec_adm.target_col, cfg=pred_cfg)

            status.update(label="Building LWBS dataset…", state="running")
            t_lwbs, spec_lwbs = build_required_target(df, which="prob_lwbs", tcfg=tcfg, pred_cfg=pred_cfg)
            lwbs_dataset = build_prediction_dataset(df, target_df=t_lwbs, target_case_col=CASE_COL, target_col=spec_lwbs.target_col, cfg=pred_cfg)

        status.update(label="Building time-to-PIA regression dataset…", state="running")
        t_pia, spec_pia = build_required_target(df, which="remaining_time_to_pia", tcfg=tcfg, pred_cfg=pred_cfg)
        pia_dataset = build_prediction_dataset(df, target_df=t_pia, target_case_col=CASE_COL, target_col=spec_pia.target_col, cfg=pred_cfg)

        mb_adm = mb_lwbs = mb_pia = None

        if admission_dataset is not None:
            status.update(label="Training admission model…", state="running")
            mb_adm = train_model(admission_dataset, task="binary", model_type=model_type_bin)

        if lwbs_dataset is not None:
            status.update(label="Training LWBS model…", state="running")
            mb_lwbs = train_model(lwbs_dataset, task="binary", model_type=model_type_bin)

        status.update(label="Training time-to-PIA regression model…", state="running")
        mb_pia = train_model(pia_dataset, task="regression", model_type=model_type_reg)

        # stash
        st.session_state["pred_cfg"] = pred_cfg
        st.session_state["tcfg"] = tcfg
        st.session_state["dataset_X_only"] = dataset_X_only
        st.session_state["X_all"] = X_all

        st.session_state["t_adm"] = t_adm
        st.session_state["t_lwbs"] = t_lwbs
        st.session_state["t_pia"] = t_pia

        st.session_state["spec_adm"] = spec_adm
        st.session_state["spec_lwbs"] = spec_lwbs
        st.session_state["spec_pia"] = spec_pia

        st.session_state["admission_dataset"] = admission_dataset
        st.session_state["lwbs_dataset"] = lwbs_dataset
        st.session_state["pia_dataset"] = pia_dataset

        st.session_state["mb_adm"] = mb_adm
        st.session_state["mb_lwbs"] = mb_lwbs
        st.session_state["mb_pia"] = mb_pia

        st.session_state["model_signature"] = sig

        status.update(label="Done.", state="complete")

    except Exception as e:
        status.update(label="Model build failed.", state="error")
        st.exception(e)
        st.stop()

# pull from session_state
pred_cfg = st.session_state["pred_cfg"]
tcfg = st.session_state["tcfg"]
X_all = st.session_state["X_all"]

t_adm = st.session_state.get("t_adm")
t_lwbs = st.session_state.get("t_lwbs")
t_pia = st.session_state.get("t_pia")

spec_adm = st.session_state.get("spec_adm")
spec_lwbs = st.session_state.get("spec_lwbs")
spec_pia = st.session_state.get("spec_pia")

admission_dataset = st.session_state.get("admission_dataset")
lwbs_dataset = st.session_state.get("lwbs_dataset")
pia_dataset = st.session_state.get("pia_dataset")

mb_adm = st.session_state.get("mb_adm")
mb_lwbs = st.session_state.get("mb_lwbs")
mb_pia = st.session_state.get("mb_pia")

# ============================================================
# Build one-row feature frame for scoring
# ============================================================
X_row = pd.DataFrame([{c: 0 for c in X_all.columns}], index=["PROFILE"])

if "n_events" in X_row.columns:
    X_row.loc["PROFILE", "n_events"] = int(n_events_so_far)
if "elapsed_minutes" in X_row.columns:
    X_row.loc["PROFILE", "elapsed_minutes"] = float(elapsed_so_far)
if "hour" in X_row.columns:
    X_row.loc["PROFILE", "hour"] = int(arrival_ts.hour)
if "dayofweek" in X_row.columns:
    X_row.loc["PROFILE", "dayofweek"] = int(arrival_ts.dayofweek)

if last_activity != "(none)":
    colname = f"last_act_{last_activity}"
    if colname in X_row.columns:
        X_row.loc["PROFILE", colname] = 1

for col, val in inputs.items():
    if col in X_row.columns:
        X_row.loc["PROFILE", col] = val
    else:
        X_row[col] = val

# ============================================================
# Historical baselines
# ============================================================
st.divider()
st.subheader("Historical baselines")

b1, b2, b3 = st.columns(3)

with b1:
    if admission_dataset is None:
        st.metric("Admission rate", "N/A")
    else:
        summ = summarize_target_with_ci(
            t_adm, case_col=CASE_COL, target_col=spec_adm.target_col, task="binary",
            alpha=0.05, n_boot=400, random_state=42
        ).iloc[0]
        rate = safe_float(summ.get("rate"))
        lo = safe_float(summ.get("rate_ci_low"))
        hi = safe_float(summ.get("rate_ci_high"))
        st.metric("Admission rate", fmt_ci(rate, lo, hi, ".3f") if show_ci else fmt_num(rate, ".3f"))

with b2:
    if lwbs_dataset is None:
        st.metric("LWBS rate", "N/A")
    else:
        summ = summarize_target_with_ci(
            t_lwbs, case_col=CASE_COL, target_col=spec_lwbs.target_col, task="binary",
            alpha=0.05, n_boot=400, random_state=42
        ).iloc[0]
        rate = safe_float(summ.get("rate"))
        lo = safe_float(summ.get("rate_ci_low"))
        hi = safe_float(summ.get("rate_ci_high"))
        st.metric("LWBS rate", fmt_ci(rate, lo, hi, ".3f") if show_ci else fmt_num(rate, ".3f"))

with b3:
    if pia_dataset is None:
        st.metric("Remaining time-to-PIA median (min)", "N/A")
    else:
        summ = summarize_target_with_ci(
            t_pia, case_col=CASE_COL, target_col=spec_pia.target_col, task="regression",
            alpha=0.05, n_boot=400, random_state=42
        ).iloc[0]
        med = safe_float(summ.get("median"))
        lo = safe_float(summ.get("median_ci_low"))
        hi = safe_float(summ.get("median_ci_high"))
        st.metric("Remaining time-to-PIA median (min)", fmt_ci(med, lo, hi, ".1f") if show_ci else fmt_num(med, ".1f"))

# ============================================================
# Predictions
# ============================================================
st.divider()
st.subheader("Predictions for this profile")

pred_cols = st.columns(3)

with pred_cols[0]:
    st.markdown("### P(Admission)")
    if mb_adm is None:
        st.info("Admission model unavailable (pick a disposition column in Configuration).")
    else:
        out = predict(mb_adm, X_row)
        p = safe_float(out["y_score"].iloc[0]) if "y_score" in out.columns else safe_float(out["y_pred"].iloc[0])

        pred_display = fmt_num(p, ".3f")
        if show_patient_ci:
            try:
                ci = bootstrap_prediction_ci(
                    df, pred_cfg=pred_cfg, tcfg=tcfg, which="prob_admission",
                    X_row=X_row, model_type=model_type_bin,
                    n_boot=int(patient_ci_boot), random_state=42
                ).iloc[0]
                lo = safe_float(ci.get("ci_low"))
                hi = safe_float(ci.get("ci_high"))
                pred_display = fmt_ci(p, lo, hi, ".3f")
            except Exception as e:
                st.warning(f"Patient CI failed: {e}")

        st.metric("Predicted probability", pred_display)

with pred_cols[1]:
    st.markdown("### P(LWBS)")
    if mb_lwbs is None:
        st.info("LWBS model unavailable (pick a disposition column in Configuration).")
    else:
        out = predict(mb_lwbs, X_row)
        p = safe_float(out["y_score"].iloc[0]) if "y_score" in out.columns else safe_float(out["y_pred"].iloc[0])

        pred_display = fmt_num(p, ".3f")
        if show_patient_ci:
            try:
                ci = bootstrap_prediction_ci(
                    df, pred_cfg=pred_cfg, tcfg=tcfg, which="prob_lwbs",
                    X_row=X_row, model_type=model_type_bin,
                    n_boot=int(patient_ci_boot), random_state=42
                ).iloc[0]
                lo = safe_float(ci.get("ci_low"))
                hi = safe_float(ci.get("ci_high"))
                pred_display = fmt_ci(p, lo, hi, ".3f")
            except Exception as e:
                st.warning(f"Patient CI failed: {e}")

        st.metric("Predicted probability", pred_display)

with pred_cols[2]:
    st.markdown("### Remaining time to PIA (min)")
    if mb_pia is None:
        st.info("Time-to-PIA model unavailable.")
    else:
        out = predict(mb_pia, X_row)
        t_pred = safe_float(out["y_pred"].iloc[0])

        pred_display = fmt_num(t_pred, ".1f")
        if show_patient_ci:
            try:
                ci = bootstrap_prediction_ci(
                    df, pred_cfg=pred_cfg, tcfg=tcfg, which="remaining_time_to_pia",
                    X_row=X_row, model_type=model_type_reg,
                    n_boot=int(patient_ci_boot), random_state=42
                ).iloc[0]
                lo = safe_float(ci.get("ci_low"))
                hi = safe_float(ci.get("ci_high"))
                pred_display = fmt_ci(t_pred, lo, hi, ".1f")
            except Exception as e:
                st.warning(f"Patient CI failed: {e}")

        st.metric("Predicted minutes", pred_display)

st.caption(
    "CI notes: historical CIs are bootstrap (and proportion CIs for admission/LWBS). "
)

# ============================================================
# Model performance summary (tables) — side-by-side
# ============================================================
st.divider()
st.subheader("Model performance summary (test split)")

left, right = st.columns([1.4, 1.0])

# ---- Left: binary table (columns include n=...)
with left:
    if admission_dataset is None and lwbs_dataset is None:
        st.info("Binary performance unavailable (needs disposition column).")
    else:
        adm_n = int(admission_dataset.y.shape[0]) if admission_dataset is not None and admission_dataset.y is not None else 0
        lwbs_n = int(lwbs_dataset.y.shape[0]) if lwbs_dataset is not None and lwbs_dataset.y is not None else 0

        adm_col = f"Admission (n={adm_n})"
        lwbs_col = f"LWBS (n={lwbs_n})"

        m_adm = {}
        m_lwbs = {}

        if admission_dataset is not None:
            ev_adm = evaluate_model(admission_dataset, task="binary", model_type=model_type_bin, test_size=0.25, random_state=42)
            m_adm = ev_adm.metrics

        if lwbs_dataset is not None:
            ev_lwbs = evaluate_model(lwbs_dataset, task="binary", model_type=model_type_bin, test_size=0.25, random_state=42)
            m_lwbs = ev_lwbs.metrics

        idx = []
        rows = []
        for k in metric_keys_binary:
            if k in m_adm or k in m_lwbs:
                idx.append(k)
                rows.append({adm_col: safe_float(m_adm.get(k)), lwbs_col: safe_float(m_lwbs.get(k))})

        bin_tbl = pd.DataFrame(rows, index=idx)
        if bin_tbl.empty:
            st.info("No binary metrics available for the selected metric set.")
        else:
            st.dataframe(bin_tbl.style.format(precision=3), use_container_width=True)

# ---- Right: regression table (column includes n=...)
with right:
    if pia_dataset is None:
        st.info("Regression performance unavailable.")
    else:
        pia_n = int(pia_dataset.y.shape[0]) if pia_dataset.y is not None else 0
        pia_col = f"Time-to-PIA (n={pia_n})"

        ev_pia = evaluate_model(pia_dataset, task="regression", model_type=model_type_reg, test_size=0.25, random_state=42)
        reg_metrics = ev_pia.metrics

        rows = []
        idx = []
        for k in metric_keys_time:
            if k in reg_metrics:
                idx.append(k)
                rows.append({pia_col: safe_float(reg_metrics.get(k))})

        reg_tbl = pd.DataFrame(rows, index=idx)
        if reg_tbl.empty:
            # fallback: show whatever exists
            fallback = pd.DataFrame({pia_col: [safe_float(v) for v in reg_metrics.values()]}, index=list(reg_metrics.keys()))
            st.dataframe(fallback.style.format(precision=3), use_container_width=True)
        else:
            st.dataframe(reg_tbl.style.format(precision=3), use_container_width=True)
