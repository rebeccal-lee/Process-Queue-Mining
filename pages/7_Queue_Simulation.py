import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import replace

from load_data import build_event_log  # adjust if needed
from simulation import (
    DEFAULT_TRANSITIONS,
    SimulationConfig,
    MCMCConfig,
    Priors,
    summarize_simulations,
    posterior_predictive_times,
    fit_weibull_aft_mcmc,
    build_transition_dataset,
    _prep_df,
)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Queue Simulation", layout="wide")
st.title("Queue Simulation")

df_raw = st.session_state.get("df_raw")
mapping = st.session_state.get("mapping")

if df_raw is None:
    st.info("Go to Home and upload a CSV first.")
    st.stop()

if mapping is None:
    st.info("Go to Upload Data and save a mapping first.")
    st.stop()

@st.cache_data(show_spinner=False)
def _build_standardized_log(df: pd.DataFrame, mapping: dict):
    event_log_df, meta = build_event_log(df, mapping)
    return event_log_df, meta

event_log_df, meta = _build_standardized_log(df_raw, mapping)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")

capacity_multiplier = st.sidebar.slider(
    "Capacity multiplier (m)",
    min_value=0.10,
    max_value=5.00,
    value=1.00,
    step=0.05,
    help="Rescales baseline posterior predictive draws: T_capacity = T_baseline / m",
)

with st.sidebar.expander("Advanced: MCMC controls", expanded=False):
    draws = st.number_input("draws", 200, 5000, 1000, 100)
    tune = st.number_input("tune", 200, 5000, 1000, 100)
    chains = st.number_input("chains", 1, 8, 2, 1)
    target_accept = st.slider("target_accept", 0.7, 0.99, 0.9, 0.01)
    seed = st.number_input("random_seed", 1, 10000, 42, 1)
    n_mc_times = st.number_input("posterior predictive draws (n_mc_times)", 500, 50000, 5000, 500)

base_cfg = SimulationConfig(
    priors=Priors(),
    mcmc=MCMCConfig(
        draws=int(draws),
        tune=int(tune),
        chains=int(chains),
        target_accept=float(target_accept),
        random_seed=int(seed),
    ),
    n_mc_times=int(n_mc_times),
)

# -----------------------------
# Hard-coded output settings
# -----------------------------
CANONICAL_STEPS = ["triage", "registration", "assessment", "consult request", "discharge"]

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

SLA_THRESHOLDS_BY_TRANSITION = {
    "triage_to_registration": (2.0, 5.0, 10.0),
    "registration_to_assessment": (10.0, 30.0, 60.0),
    "assessment_to_consult_request": (30.0, 60.0, 120.0),
    "assessment_to_discharge": (60.0, 120.0, 240.0),
}

def _cfg_for_transition(cfg: SimulationConfig, transition_key: str) -> SimulationConfig:
    slas = SLA_THRESHOLDS_BY_TRANSITION.get(transition_key, (5.0, 10.0, 30.0))
    return replace(cfg, summary_quantiles=QUANTILES, sla_thresholds_minutes=slas)

def canonicalize_activities(df: pd.DataFrame, *, act_col: str, dataset_to_canonical: dict[str, str]) -> pd.DataFrame:
    """
    dataset_to_canonical: {dataset_activity_value: canonical_label}
    Returns a copy with activity rewritten to canonical labels for mapped rows.
    Unmapped activities are left as lowercased strings.
    """
    out = df.copy()
    raw = out[act_col].astype(str).str.strip()
    norm = raw.str.lower()

    map_norm = {str(k).strip().lower(): v for k, v in dataset_to_canonical.items()}
    out[act_col] = norm.map(map_norm).fillna(norm)
    return out

def _fingerprint_event_log(df: pd.DataFrame, cfg: SimulationConfig) -> str:
    core = df[[cfg.case_col, cfg.act_col, cfg.time_col]].copy()
    core[cfg.time_col] = pd.to_datetime(core[cfg.time_col], errors="coerce")
    core = core.dropna()
    h = pd.util.hash_pandas_object(core, index=False).values
    return hashlib.sha256(h.tobytes()).hexdigest()[:16]

def _fit_key(dataset_fp: str, transition_key: str, cfg: SimulationConfig) -> str:
    return "|".join([
        dataset_fp,
        transition_key,
        str(cfg.mcmc.draws),
        str(cfg.mcmc.tune),
        str(cfg.mcmc.chains),
        f"{cfg.mcmc.target_accept:.3f}",
        str(cfg.mcmc.random_seed),
        str(cfg.n_mc_times),
        ",".join(map(str, cfg.summary_quantiles)),
        ",".join(map(str, cfg.sla_thresholds_minutes)),
        ",".join(cfg.covariate_cols) if cfg.covariate_cols else "",
    ])

def _quantile_table(x: np.ndarray, qs: tuple[float, ...]) -> pd.DataFrame:
    qvals = np.quantile(x, list(qs))
    return pd.DataFrame({
        "quantile": [f"p{int(round(q*100)):02d}" for q in qs],
        "minutes": qvals.astype(float),
    })

def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([]), np.array([])
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys

# -----------------------------
# TOP OF PAGE: Activity mapping UI
# -----------------------------
st.subheader("1) Map your dataset activities")

raw_acts = (
    event_log_df[base_cfg.act_col]
    .astype(str)
    .str.strip()
    .dropna()
    .unique()
    .tolist()
)
raw_acts_sorted = sorted(raw_acts, key=lambda s: s.lower())

map_state_key = "activity_mapping_canonical_to_dataset"
if map_state_key not in st.session_state:
    st.session_state[map_state_key] = {c: None for c in CANONICAL_STEPS}

canonical_to_dataset = st.session_state[map_state_key]

cols = st.columns(2)
for i, canon in enumerate(CANONICAL_STEPS):
    with cols[i % 2]:
        canonical_to_dataset[canon] = st.selectbox(
            f"Map **{canon}** to a dataset activity value",
            options=[None] + raw_acts_sorted,
            index=0 if canonical_to_dataset.get(canon) is None else (raw_acts_sorted.index(canonical_to_dataset[canon]) + 1),
            key=f"map_{canon}",
        )

# Check for duplicate dataset values mapped to multiple canonical steps
chosen_vals = [v for v in canonical_to_dataset.values() if v is not None]
dupes = sorted({v for v in chosen_vals if chosen_vals.count(v) > 1}, key=lambda s: s.lower())
if dupes:
    st.warning(
        "You mapped the same dataset activity value to multiple canonical steps: "
        + ", ".join(repr(d) for d in dupes)
        + ". This can create ambiguous transitions."
    )

st.session_state[map_state_key] = canonical_to_dataset

dataset_to_canonical = {v: k for k, v in canonical_to_dataset.items() if v is not None}

# Apply mapping + standard prep
df_mapped = canonicalize_activities(event_log_df, act_col=base_cfg.act_col, dataset_to_canonical=dataset_to_canonical)
df_for_model = _prep_df(df_mapped, base_cfg)  # parses timestamps, lowercases, sorts

# -----------------------------
# Select transitions (multiselect)
# -----------------------------
st.subheader("2) Select transitions")

label_map = {t.key: f"{t.start_activity} → {t.end_activity}" for t in DEFAULT_TRANSITIONS}

# Only allow transitions whose endpoints are mapped (optional but prevents n=0 confusion)
def _is_mapped(canon_step: str) -> bool:
    return canonical_to_dataset.get(canon_step) is not None

eligible_keys = []
for t in DEFAULT_TRANSITIONS:
    if _is_mapped(t.start_activity) and _is_mapped(t.end_activity):
        eligible_keys.append(t.key)

if not eligible_keys:
    st.info("Map at least two steps (start and end) to enable a transition.")
    st.stop()

selected_keys = st.multiselect(
    "Choose one or more transitions",
    options=eligible_keys,
    default=[eligible_keys[0]],
    format_func=lambda k: f"{label_map.get(k, k)} ({k})",
)

run_model = st.button("Run model", type="primary")

# -----------------------------
# Cache and run gating
# -----------------------------
fit_state = st.session_state.setdefault("sim_fit_state", {})

dataset_fp = _fingerprint_event_log(df_for_model, base_cfg)

if not selected_keys:
    st.info("Select at least one transition.")
    st.stop()

# If they haven't clicked run and nothing cached for these selections, do nothing
have_any_cached = any(_fit_key(dataset_fp, k, _cfg_for_transition(base_cfg, k)) in fit_state for k in selected_keys)
if not run_model and not have_any_cached:
    st.info("Click **Run model** to fit the selected transition(s).")
    st.stop()

# -----------------------------
# Fit models (only when needed)
# -----------------------------
if run_model:
    with st.spinner("Running model — this could take a couple of minutes..."):
        for transition_key in selected_keys:
            cfg_t = _cfg_for_transition(base_cfg, transition_key)
            key = _fit_key(dataset_fp, transition_key, cfg_t)

            # Cache behavior: only fit if not already cached for this dataset+settings.
            # If you want to ALWAYS refit on every click, set need_fit = True.
            need_fit = key not in fit_state

            if not need_fit:
                continue

            spec = next(t for t in DEFAULT_TRANSITIONS if t.key == transition_key)

            y_time, y_event, X, used_case_ids = build_transition_dataset(
                df_for_model,
                start_activity=spec.start_activity,
                end_activity=spec.end_activity,
                cfg=cfg_t,
            )

            if len(y_time) < 10:
                start_ct = int((df_for_model[cfg_t.act_col] == spec.start_activity).sum())
                end_ct = int((df_for_model[cfg_t.act_col] == spec.end_activity).sum())
                st.error(
                    f"Too few cases for {transition_key!r} (n={len(y_time)}). "
                    f"Counts after mapping: start '{spec.start_activity}' rows={start_ct}, "
                    f"end '{spec.end_activity}' rows={end_ct}."
                )
                continue

            posterior = fit_weibull_aft_mcmc(
                y_time=y_time,
                y_event=y_event,
                X=X,
                cfg=cfg_t,
            )

            X_pred = np.zeros((X.shape[1],), dtype=float)
            sim_base = posterior_predictive_times(
                posterior_draws=posterior,
                X_pred=X_pred,
                n_times=cfg_t.n_mc_times,
                cfg=cfg_t,
            )

            fit_state[key] = {
                "posterior": posterior,
                "sim_base": sim_base,
                "n_cases_used": int(len(y_time)),
                "cfg": cfg_t,
            }

        st.session_state["sim_fit_state"] = fit_state

# -----------------------------
# Results
# -----------------------------
st.divider()
st.subheader("Results")

def _mapped_from(canon_step: str) -> str:
    v = canonical_to_dataset.get(canon_step)
    return f" (mapped from '{v}')" if v else ""

for transition_key in selected_keys:
    cfg_t = _cfg_for_transition(base_cfg, transition_key)
    key = _fit_key(dataset_fp, transition_key, cfg_t)

    if key not in fit_state:
        continue

    spec = next(t for t in DEFAULT_TRANSITIONS if t.key == transition_key)

    title = (
        f"{spec.start_activity}{_mapped_from(spec.start_activity)} "
        f"→ {spec.end_activity}{_mapped_from(spec.end_activity)} "
        f"({transition_key})"
    )
    st.markdown(f"## {title}")

    posterior = fit_state[key]["posterior"]
    sim_base = fit_state[key]["sim_base"]
    n_cases_used = fit_state[key]["n_cases_used"]

    sim_cap = sim_base / float(capacity_multiplier)

    # Summary + SLA
    summary = summarize_simulations(
        sim_base,
        sim_cap,
        cfg=cfg_t,
        capacity_multiplier=float(capacity_multiplier),
        transition_key=transition_key,
        n_cases_used=n_cases_used,
    )

    st.caption(
        f"Capacity m = {capacity_multiplier:.2f} | "
        f"SLA thresholds: {', '.join(str(int(x)) for x in cfg_t.sla_thresholds_minutes)} minutes | "
        f"n_cases_used = {n_cases_used}"
    )

    # Quantiles table
    q_base = _quantile_table(sim_base, cfg_t.summary_quantiles).rename(columns={"minutes": "baseline_minutes"})
    q_cap = _quantile_table(sim_cap, cfg_t.summary_quantiles).rename(columns={"minutes": "capacity_minutes"})
    q_tbl = q_base.merge(q_cap, on="quantile")
    st.dataframe(q_tbl, use_container_width=True)

    # SLA table (compact)
    sla_cols = ["transition_key", "capacity_multiplier", "n_cases_used"] + [
        c for c in summary.columns if c.startswith("baseline_p_le_") or c.startswith("capacity_p_le_")
    ]
    st.dataframe(summary[sla_cols], use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Distribution")
        st.caption("Histogram distribution of wait times.")
        fig = plt.figure()
        # Use same x-limits so comparison is visually fair
        xmax = float(np.quantile(sim_base, 0.99))
        xmax = max(xmax, float(np.quantile(sim_cap, 0.99)))
        xmax = max(xmax, 1e-6)

        plt.hist(sim_base, bins=50, alpha=0.6, label="Baseline")
        plt.hist(sim_cap, bins=50, alpha=0.6, label="Capacity")
        plt.xlim(0, xmax)
        plt.xlabel("Minutes")
        plt.ylabel("Count")
        plt.legend()
        st.pyplot(fig, clear_figure=True)

    with c2:
        st.subheader("CDF")
        st.caption("Cumulative probability of obtaining a wait time less than or equal to a given wait time.")
        fig2 = plt.figure()

        xb, yb = _ecdf(sim_base)
        xc, yc = _ecdf(sim_cap)

        # same x-range so "capacity shifts left" is obvious
        xmax = float(np.quantile(sim_base, 0.99))
        xmax = max(xmax, float(np.quantile(sim_cap, 0.99)))
        xmax = max(xmax, 1e-6)

        if len(xb) > 0:
            plt.step(xb, yb, where="post", label="Baseline")
        if len(xc) > 0:
            plt.step(xc, yc, where="post", label="Capacity")

        plt.xlim(0, xmax)
        plt.ylim(0, 1)
        plt.xlabel("Minutes")
        plt.ylabel("P(T ≤ t)")
        plt.legend()
        st.pyplot(fig2, clear_figure=True)

    # Downloads
    st.download_button(
        f"Download quantiles ({transition_key}).csv",
        data=q_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"quantiles_{transition_key}.csv",
        mime="text/csv",
        key=f"dl_q_{transition_key}",
    )
    st.download_button(
        f"Download posterior_draws ({transition_key}).csv",
        data=posterior.to_csv(index=False).encode("utf-8"),
        file_name=f"posterior_draws_{transition_key}.csv",
        mime="text/csv",
        key=f"dl_post_{transition_key}",
    )

    st.divider()
