# pages/5_Prediction.py
from __future__ import annotations

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
)

st.set_page_config(page_title="Patient Prediction", layout="wide")
st.title("Patient Prediction")
st.caption("Predict admission risk, LWBS risk, and remaining time-to-PIA.")

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
# Session defaults
# -----------------------------
st.session_state.setdefault("score_result", None)
st.session_state.setdefault("pred_show_ci", False)
st.session_state.setdefault("pred_n_boot", 2000)
st.session_state.setdefault("pred_n_refits", 200)

# -----------------------------
# Helpers
# -----------------------------
def bootstrap_ci(values: np.ndarray, stat_fn, n_boot: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (np.nan, np.nan)
    n = values.size
    stats = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        samp = rng.choice(values, size=n, replace=True)
        stats[i] = float(stat_fn(samp))
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))


def fmt_ci(val: float, lo: float, hi: float, decimals: int = 3) -> str:
    if not (np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi)):
        return f"{val:.{decimals}f}" if np.isfinite(val) else "N/A"
    return f"{val:.{decimals}f} ({lo:.{decimals}f}–{hi:.{decimals}f})"


def fmt_ci_min(val: float, lo: float, hi: float, decimals: int = 1) -> str:
    if not (np.isfinite(val) and np.isfinite(lo) and np.isfinite(hi)):
        return f"{val:.{decimals}f}" if np.isfinite(val) else "N/A"
    return f"{val:.{decimals}f} ({lo:.{decimals}f}–{hi:.{decimals}f})"


def align_dataset_y_to_X(ds):
    if ds is None or ds.y is None:
        return ds
    y = ds.y.copy()
    if not y.index.equals(ds.X.index):
        y.index = ds.X.index
    return type(ds)(X=ds.X, y=y, case_index=ds.case_index, feature_info=ds.feature_info, cfg=ds.cfg)


def bootstrap_refit_pred_ci(
    ds,
    *,
    task: str,
    model_type: str,
    X_row: pd.DataFrame,
    score_col: str,
    n_refits: int,
    random_state: int = 42,
):
    if ds is None or ds.y is None:
        return (np.nan, np.nan)
    ds = align_dataset_y_to_X(ds)
    idx = np.array(ds.X.index)
    if idx.size < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(int(random_state))
    preds = np.empty(int(n_refits), dtype=float)

    for b in range(int(n_refits)):
        samp = rng.choice(idx, size=idx.size, replace=True)
        Xb = ds.X.loc[samp]
        yb = ds.y.loc[samp]
        ds_b = type(ds)(X=Xb, y=yb, case_index=Xb.index, feature_info=ds.feature_info, cfg=ds.cfg)

        mb = train_model(ds_b, task=task, model_type=model_type)
        po = predict(mb, X_row.set_index(pd.Index(["PROFILE"])))
        preds[b] = float(po[score_col].iloc[0])

    return (float(np.quantile(preds, 0.025)), float(np.quantile(preds, 0.975)))


# ============================================================
# Patient profile + config + button (ONE form)
# Nothing heavy runs until form submit.
# ============================================================
st.subheader("Patient profile")
st.caption("Set configuration + profile inputs, then click the button to score.")

with st.form("patient_profile_form", clear_on_submit=False):

    # -----------------------------
    # Configuration expander
    # -----------------------------
    with st.expander("Configuration", expanded=False):
        c1, c2, c3 = st.columns([1.2, 1.2, 1.0])

        with c1:
            pia_activity = st.selectbox(
                "Define PIA activity (first provider assessment event)",
                options=unique_acts,
                index=0,
            )
            cutoff_minutes = st.slider(
                "Prediction cutoff window (minutes since arrival)",
                min_value=0,
                max_value=240,
                value=30,
                step=5,
            )

        with c2:
            candidate_static = [c for c in df.columns if c not in [CASE_COL, TIME_COL, ACT_COL]]
            default_static = [c for c in ["age", "triage_code", "triage_desc", "initial_zone", "gender"] if c in candidate_static]

            static_feature_cols = st.multiselect(
                "Profile fields (static features)",
                options=candidate_static,
                default=default_static,
            )

            disposition_col_guess = "disposition_desc" if "disposition_desc" in df.columns else None
            disposition_col = st.selectbox(
                "Disposition column (for admission/LWBS targets)",
                options=["(none)"] + candidate_static,
                index=(1 + candidate_static.index(disposition_col_guess)) if disposition_col_guess in candidate_static else 0,
            )

        with c3:
            st.markdown("**Model options**")
            model_type = st.selectbox("Model type (binary targets)", options=["logreg", "rf"], index=0)
            st.caption("CI settings are below (applies instantly).")

    # -----------------------------
    # Profile inputs
    # -----------------------------
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
            value=min(30, int(cutoff_minutes)),
            step=5,
        )

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
                inputs[col] = st.selectbox(col, options=opts, index=0) if opts else st.text_input(col, value="")

    # ✅ Button is under the options / profile inputs, inside the same section
    do_score = st.form_submit_button("Score this patient", type="primary")


# ============================================================
# CI controls OUTSIDE the form (so they update immediately)
# ============================================================
with st.expander("Uncertainty (confidence intervals)", expanded=False):
    st.checkbox("Show 95% CI (bootstrap)", key="pred_show_ci")

    st.number_input(
        "Baseline bootstrap resamples (n)",
        min_value=200,
        max_value=20000,
        step=200,
        key="pred_n_boot",
        help="Used for historical baseline CIs.",
    )
    st.number_input(
        "Prediction CI refits (n)",
        min_value=10,
        max_value=2000,
        step=10,
        key="pred_n_refits",
        help="Used for profile prediction CIs (refit bootstrap).",
    )

# ============================================================
# Heavy work ONLY runs on button click
# ============================================================
if do_score:
    status = st.status("Scoring patient…", expanded=False)
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

        admission_dataset = None
        lwbs_dataset = None
        admission_available = tcfg.disposition_col is not None

        if admission_available:
            status.update(label="Building admission target…", state="running")
            t_adm, spec_adm = build_required_target(df, which="prob_admission", tcfg=tcfg, pred_cfg=pred_cfg)
            admission_dataset = build_prediction_dataset(
                df, target_df=t_adm, target_case_col=CASE_COL, target_col=spec_adm.target_col, cfg=pred_cfg
            )
            admission_dataset = align_dataset_y_to_X(admission_dataset)

            status.update(label="Building LWBS target…", state="running")
            t_lwbs, spec_lwbs = build_required_target(df, which="prob_lwbs", tcfg=tcfg, pred_cfg=pred_cfg)
            lwbs_dataset = build_prediction_dataset(
                df, target_df=t_lwbs, target_case_col=CASE_COL, target_col=spec_lwbs.target_col, cfg=pred_cfg
            )
            lwbs_dataset = align_dataset_y_to_X(lwbs_dataset)

        status.update(label="Building remaining time-to-PIA target…", state="running")
        t_pia, spec_pia = build_required_target(df, which="remaining_time_to_pia", tcfg=tcfg, pred_cfg=pred_cfg)
        pia_dataset = build_prediction_dataset(
            df, target_df=t_pia, target_case_col=CASE_COL, target_col=spec_pia.target_col, cfg=pred_cfg
        )
        pia_dataset = align_dataset_y_to_X(pia_dataset)

        status.update(label="Training models…", state="running")
        mb_adm = train_model(admission_dataset, task="binary", model_type=model_type) if admission_dataset is not None else None
        mb_lwbs = train_model(lwbs_dataset, task="binary", model_type=model_type) if lwbs_dataset is not None else None
        mb_pia = train_model(pia_dataset, task="regression", model_type="rf")

        status.update(label="Building one-row profile…", state="running")
        X_row = pd.DataFrame([{c: 0 for c in X_all.columns}], index=["PROFILE"])

        if "n_events" in X_row.columns:
            X_row.loc["PROFILE", "n_events"] = int(n_events_so_far)
        if "elapsed_minutes" in X_row.columns:
            X_row.loc["PROFILE", "elapsed_minutes"] = float(elapsed_so_far)
        if "hour" in X_row.columns:
            X_row.loc["PROFILE", "hour"] = int(arrival_ts.hour)
        if "dayofweek" in X_row.columns:
            X_row.loc["PROFILE", "dayofweek"] = int(arrival_ts.dayofweek)

        for col, val in inputs.items():
            if col in X_row.columns:
                X_row.loc["PROFILE", col] = val
            else:
                X_row[col] = val

        # store everything needed for display (INCLUDING trained models so reruns don't retrain)
        st.session_state["score_result"] = dict(
            admission_dataset=admission_dataset,
            lwbs_dataset=lwbs_dataset,
            pia_dataset=pia_dataset,
            X_row=X_row,
            model_type=model_type,
            mb_adm=mb_adm,
            mb_lwbs=mb_lwbs,
            mb_pia=mb_pia,
        )

        status.update(label="Done.", state="complete")

    except Exception as e:
        status.update(label="Failed.", state="error")
        st.exception(e)

# ============================================================
# Display (only if score_result exists)
# ============================================================
score_pack = st.session_state.get("score_result")
if score_pack is None:
    st.info("Fill the profile above and click **Score this patient**.")
    st.stop()

admission_dataset = score_pack["admission_dataset"]
lwbs_dataset = score_pack["lwbs_dataset"]
pia_dataset = score_pack["pia_dataset"]
X_row = score_pack["X_row"]
model_type = score_pack["model_type"]

mb_adm = score_pack.get("mb_adm")
mb_lwbs = score_pack.get("mb_lwbs")
mb_pia = score_pack.get("mb_pia")

show_bootstrap_ci = bool(st.session_state.get("pred_show_ci", False))
n_boot = int(st.session_state.get("pred_n_boot", 1000))
n_refits = int(st.session_state.get("pred_n_refits", 100))

# ---- Historical baselines ----
st.divider()
st.subheader("Historical baselines")
b1, b2, b3 = st.columns(3)

if admission_dataset is None:
    b1.metric("Admission rate", "N/A")
else:
    y = pd.to_numeric(admission_dataset.y, errors="coerce").dropna().to_numpy()
    rate = float(np.mean(y)) if len(y) else np.nan
    if show_bootstrap_ci:
        lo, hi = bootstrap_ci(y, np.mean, n_boot=n_boot)
        b1.metric("Admission rate", fmt_ci(rate, lo, hi, decimals=3))
    else:
        b1.metric("Admission rate", f"{rate:.3f}")

if lwbs_dataset is None:
    b2.metric("LWBS rate", "N/A")
else:
    y = pd.to_numeric(lwbs_dataset.y, errors="coerce").dropna().to_numpy()
    rate = float(np.mean(y)) if len(y) else np.nan
    if show_bootstrap_ci:
        lo, hi = bootstrap_ci(y, np.mean, n_boot=n_boot)
        b2.metric("LWBS rate", fmt_ci(rate, lo, hi, decimals=3))
    else:
        b2.metric("LWBS rate", f"{rate:.3f}")

tvals = pd.to_numeric(pia_dataset.y, errors="coerce").dropna().to_numpy()
med = float(np.median(tvals)) if len(tvals) else np.nan
if show_bootstrap_ci:
    lo, hi = bootstrap_ci(tvals, np.median, n_boot=n_boot)
    b3.metric("Remaining time-to-PIA median (min)", fmt_ci_min(med, lo, hi, decimals=1))
else:
    b3.metric("Remaining time-to-PIA median (min)", f"{med:.1f}")

# ---- Predictions ----
st.divider()
st.subheader("Predictions for this profile")
pred_cols = st.columns(3)

with pred_cols[0]:
    st.markdown("#### P(Admission)")
    if mb_adm is None:
        st.info("Admission model unavailable (pick a disposition column).")
    else:
        out = predict(mb_adm, X_row.set_index(pd.Index(["PROFILE"])))
        p = float(out["y_score"].iloc[0]) if "y_score" in out.columns else float(out["y_pred"].iloc[0])
        if show_bootstrap_ci:
            lo, hi = bootstrap_refit_pred_ci(
                admission_dataset,
                task="binary",
                model_type=model_type,
                X_row=X_row,
                score_col="y_score" if "y_score" in out.columns else "y_pred",
                n_refits=n_refits,
                random_state=101,
            )
            st.metric("Predicted probability", fmt_ci(p, lo, hi, decimals=3))
        else:
            st.metric("Predicted probability", f"{p:.3f}")

with pred_cols[1]:
    st.markdown("#### P(LWBS)")
    if mb_lwbs is None:
        st.info("LWBS model unavailable (pick a disposition column).")
    else:
        out = predict(mb_lwbs, X_row.set_index(pd.Index(["PROFILE"])))
        p = float(out["y_score"].iloc[0]) if "y_score" in out.columns else float(out["y_pred"].iloc[0])
        if show_bootstrap_ci:
            lo, hi = bootstrap_refit_pred_ci(
                lwbs_dataset,
                task="binary",
                model_type=model_type,
                X_row=X_row,
                score_col="y_score" if "y_score" in out.columns else "y_pred",
                n_refits=n_refits,
                random_state=202,
            )
            st.metric("Predicted probability", fmt_ci(p, lo, hi, decimals=3))
        else:
            st.metric("Predicted probability", f"{p:.3f}")

with pred_cols[2]:
    st.markdown("#### Remaining time to PIA (min)")
    out = predict(mb_pia, X_row.set_index(pd.Index(["PROFILE"])))
    yhat = float(out["y_pred"].iloc[0])
    if show_bootstrap_ci:
        lo, hi = bootstrap_refit_pred_ci(
            pia_dataset,
            task="regression",
            model_type="rf",
            X_row=X_row,
            score_col="y_pred",
            n_refits=n_refits,
            random_state=303,
        )
        st.metric("Predicted minutes", fmt_ci_min(yhat, lo, hi, decimals=1))
    else:
        st.metric("Predicted minutes", f"{yhat:.1f}")

# ---- Performance summary ----
st.divider()
st.subheader("Model performance summary")
perf_mode = st.selectbox("Performance mode", options=["Binary", "Time (regression)"], index=0)

if perf_mode == "Binary":
    rows = ["n_test", "accuracy", "f1", "precision", "recall", "roc_auc", "avg_precision"]
    perf = pd.DataFrame(index=rows)

    def add_binary(ds, colname: str):
        if ds is None:
            perf[colname] = np.nan
            perf.loc["n_test", colname] = 0
            return
        ev = evaluate_model(ds, task="binary", model_type=model_type, test_size=0.25, random_state=42)
        per = ev.per_case.copy()
        metrics = dict(ev.metrics)
        perf[colname] = np.nan
        perf.loc["n_test", colname] = int(len(per))

        try:
            from sklearn.metrics import precision_score, recall_score
            y_true = per["y_true"].astype(int).values
            y_pred = per["y_pred"].astype(int).values
            metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        except Exception:
            pass

        for k, v in metrics.items():
            if k in perf.index:
                perf.loc[k, colname] = float(v)

    add_binary(admission_dataset, f"Admission (n={admission_dataset.X.shape[0] if admission_dataset is not None else 0})")
    add_binary(lwbs_dataset, f"LWBS (n={lwbs_dataset.X.shape[0] if lwbs_dataset is not None else 0})")
    st.dataframe(perf, use_container_width=True)
else:
    ev = evaluate_model(pia_dataset, task="regression", model_type="rf", test_size=0.25, random_state=42)
    tperf = pd.DataFrame({f"Remaining time-to-PIA (n={pia_dataset.X.shape[0]})": ev.metrics})
    st.dataframe(tperf, use_container_width=True)
