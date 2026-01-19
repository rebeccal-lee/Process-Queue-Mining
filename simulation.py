
# simulation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Public data structures
# ============================================================

@dataclass(frozen=True)
class TransitionSpec:
    """A -> B transition (time from first A to first B after A)."""
    key: str
    start_activity: str
    end_activity: str


# Canonical transitions you said the UI will offer
DEFAULT_TRANSITIONS: Tuple[TransitionSpec, ...] = (
    TransitionSpec("triage_to_registration", "triage", "registration"),
    TransitionSpec("registration_to_assessment", "registration", "assessment"),
    TransitionSpec("assessment_to_consult_request", "assessment", "consult request"),
    TransitionSpec("assessment_to_discharge", "assessment", "discharge"),
)


@dataclass(frozen=True)
class Priors:
    """
    Priors for Weibull AFT.

    NOTE: In a Weibull AFT model, dispersion sigma and Weibull shape kappa are linked.
    A common parameterization is:
        log(T) = mu + sigma * Gumbel(0,1)
      => T ~ Weibull(shape=kappa=1/sigma, scale=exp(mu))

    To keep things identifiable, we treat sigma as the primary parameter and define:
        kappa = 1/sigma

    If you prefer specifying kappa directly, set use_kappa_prior=True (then sigma=1/kappa).
    """
    # Intercept prior (mu on log-time scale). If auto_center=True, mu is centered on log(median observed time).
    mu_mean: float = 0.0
    mu_sd: float = 2.0
    auto_center: bool = True

    # Optional covariate priors (beta ~ Normal(0, beta_sd))
    beta_sd: float = 1.0

    # Shape/dispersion priors (pick ONE to be free; the other becomes deterministic)
    use_kappa_prior: bool = False
    kappa_gamma_shape: float = 2.0
    kappa_gamma_rate: float = 2.0  # mean = shape/rate

    sigma_exponential_rate: float = 1.0  # mean = 1/rate


@dataclass(frozen=True)
class MCMCConfig:
    draws: int = 1000
    tune: int = 1000
    chains: int = 2
    target_accept: float = 0.9
    random_seed: int = 42


@dataclass(frozen=True)
class SimulationConfig:
    # Standardized event-log schema
    case_col: str = "case_id"
    time_col: str = "timestamp"
    act_col: str = "activity"

    # Normalization (so you can assume canonical labels later)
    normalize_activities: bool = True

    # Optional covariates (must be in event_log_df if used)
    covariate_cols: Optional[Sequence[str]] = None

    # Censoring and trace handling
    sort_events: bool = True
    require_start_activity: bool = True  # if False, cases without start are ignored anyway
    min_positive_duration: float = 1e-6  # avoid 0 durations

    # Priors + MCMC
    priors: Priors = Priors()
    mcmc: MCMCConfig = MCMCConfig()

    # Posterior predictive simulation
    n_mc_times: int = 5000  # number of survival times to draw from posterior predictive mixture
    summary_quantiles: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95)
    sla_thresholds_minutes: Tuple[float, ...] = (30.0, 60.0, 120.0)


@dataclass(frozen=True)
class SimulationResult:
    transition_key: str
    capacity_multiplier: float
    n_cases_used: int
    summary: pd.DataFrame                 # one-row df with summary stats (baseline + capacity)
    posterior_draws: pd.DataFrame         # tidy draws for key parameters
    simulated_times_baseline: np.ndarray  # posterior predictive mixture draws
    simulated_times_capacity: np.ndarray  # scaled by T/m (or via exponent form if added later)


# ============================================================
# UI-friendly helpers
# ============================================================

def list_available_transitions(
    event_log_df: pd.DataFrame,
    *,
    transitions: Sequence[TransitionSpec] = DEFAULT_TRANSITIONS,
    cfg: SimulationConfig = SimulationConfig(),
) -> pd.DataFrame:
    """
    Returns a small table indicating which canonical transitions are available
    (i.e., both start and end activities appear in the data).
    """
    df = _prep_df(event_log_df, cfg)
    acts = set(df[cfg.act_col].astype(str).tolist())
    rows = []
    for t in transitions:
        rows.append({
            "transition_key": t.key,
            "start_activity": t.start_activity,
            "end_activity": t.end_activity,
            "start_present": t.start_activity in acts,
            "end_present": t.end_activity in acts,
            "available": (t.start_activity in acts) and (t.end_activity in acts),
        })
    return pd.DataFrame(rows)


# ============================================================
# Main entrypoint
# ============================================================

def simulate_transition(
    event_log_df: pd.DataFrame,
    *,
    transition_key: str,
    capacity_multiplier: float = 1.0,
    transitions: Sequence[TransitionSpec] = DEFAULT_TRANSITIONS,
    cfg: SimulationConfig = SimulationConfig(),
    covariate_values: Optional[Dict[str, Any]] = None,
) -> SimulationResult:
    """
    Fit a Bayesian Weibull AFT model for one transition and run posterior predictive simulation.
    Then apply capacity scaling: T_capacity = T_baseline / m

    Assumptions:
      - activities are already canonical labels like "registration", "assessment", etc
        (your UI will handle mapping later)
      - event_log_df is your standardized log with columns: case_id, activity, timestamp (+ optional covariates)

    Returns:
      - summary stats for baseline and capacity scenario
      - posterior draws (mu, betas, sigma, kappa)
      - simulated times arrays
    """
    if capacity_multiplier <= 0:
        raise ValueError("capacity_multiplier must be > 0")

    spec = _get_transition_spec(transition_key, transitions)

    df = _prep_df(event_log_df, cfg)

    # Build (time, event) arrays for this A->B transition
    y_time, y_event, X, used_case_ids = build_transition_dataset(
        df,
        start_activity=spec.start_activity,
        end_activity=spec.end_activity,
        cfg=cfg,
        covariate_values=covariate_values,
    )

    if len(y_time) < 10:
        raise ValueError(
            f"Too few cases for transition {transition_key!r} "
            f"(n={len(y_time)}). Check labels and data coverage."
        )

    # Fit Bayesian Weibull AFT via MCMC
    posterior = fit_weibull_aft_mcmc(
        y_time=y_time,
        y_event=y_event,
        X=X,
        cfg=cfg,
    )

    # Posterior predictive mixture simulation of survival times
    sim_base = posterior_predictive_times(
        posterior_draws=posterior,
        X_pred=_make_X_pred(cfg, covariate_values),
        n_times=cfg.n_mc_times,
        cfg=cfg,
    )

    sim_cap = sim_base / float(capacity_multiplier)

    summary = summarize_simulations(
        sim_base,
        sim_cap,
        cfg=cfg,
        capacity_multiplier=capacity_multiplier,
        transition_key=transition_key,
        n_cases_used=len(y_time),
    )

    return SimulationResult(
        transition_key=transition_key,
        capacity_multiplier=float(capacity_multiplier),
        n_cases_used=len(y_time),
        summary=summary,
        posterior_draws=posterior,
        simulated_times_baseline=sim_base,
        simulated_times_capacity=sim_cap,
    )


# ============================================================
# Dataset building: A->B transition times with right-censoring
# ============================================================

def build_transition_dataset(
    event_log_df: pd.DataFrame,
    *,
    start_activity: str,
    end_activity: str,
    cfg: SimulationConfig,
    covariate_values: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_time:  (n,) durations in minutes (min positive enforced)
      y_event: (n,) 1 if observed end_activity, 0 if right-censored
      X:       (n,p) covariate matrix (p may be 0)
      case_ids:(n,)  case ids used
    """
    case = cfg.case_col
    tcol = cfg.time_col
    acol = cfg.act_col

    required = {case, tcol, acol}
    missing = required - set(event_log_df.columns)
    if missing:
        raise KeyError(f"event_log_df missing required columns: {missing}. Found: {list(event_log_df.columns)}")

    cov_cols = list(cfg.covariate_cols) if cfg.covariate_cols else []

    # For covariates: we treat them as case-level static features
    # (take first non-null value within each case).
    case_cov = None
    if cov_cols:
        for c in cov_cols:
            if c not in event_log_df.columns:
                raise KeyError(f"covariate_col {c!r} not found in event_log_df columns.")
        case_cov = (
            event_log_df[[case] + cov_cols]
            .groupby(case, as_index=True)
            .first()
        )

    y_time: list[float] = []
    y_event: list[int] = []
    X_rows: list[np.ndarray] = []
    used_cases: list[Any] = []

    for cid, g in event_log_df.groupby(case, sort=False):
        acts = g[acol].astype(str).tolist()
        times = g[tcol].tolist()

        # locate first start
        try:
            i_start = acts.index(start_activity)
        except ValueError:
            if cfg.require_start_activity:
                continue
            else:
                continue

        t0 = times[i_start]

        # find first end AFTER start
        t_end = None
        for j in range(i_start + 1, len(acts)):
            if acts[j] == end_activity:
                t_end = times[j]
                break

        t_last = times[-1]

        if t_end is not None:
            dt = (t_end - t0).total_seconds() / 60.0
            event = 1
        else:
            dt = (t_last - t0).total_seconds() / 60.0
            event = 0

        if not np.isfinite(dt):
            continue

        dt = float(max(dt, cfg.min_positive_duration))

        # covariates
        if cov_cols:
            row = case_cov.loc[cid].to_numpy(dtype=float, copy=True)
        else:
            row = np.zeros((0,), dtype=float)

        y_time.append(dt)
        y_event.append(event)
        X_rows.append(row)
        used_cases.append(cid)

    y_time_arr = np.asarray(y_time, dtype=float)
    y_event_arr = np.asarray(y_event, dtype=int)
    X_arr = np.vstack(X_rows) if X_rows and (len(X_rows[0]) > 0) else np.zeros((len(y_time_arr), 0), dtype=float)
    case_ids_arr = np.asarray(used_cases)

    return y_time_arr, y_event_arr, X_arr, case_ids_arr


# ============================================================
# Bayesian Weibull AFT via PyMC
# ============================================================

def fit_weibull_aft_mcmc(
    *,
    y_time: np.ndarray,
    y_event: np.ndarray,
    X: np.ndarray,
    cfg: SimulationConfig,
) -> pd.DataFrame:
    """
    Fits a Weibull AFT model with right-censoring using PyMC.

    Returns a tidy DataFrame of posterior draws for:
      - mu (intercept)
      - beta_* (if covariates)
      - sigma
      - kappa (= 1/sigma) OR (if use_kappa_prior=True) kappa and sigma (=1/kappa)

    Dependency:
      pip install pymc arviz
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError as e:
        raise ImportError(
            "PyMC + ArviZ are required for Bayesian MCMC. Install with:\n"
            "  pip install pymc arviz"
        ) from e

    y_time = np.asarray(y_time, dtype=float)
    y_event = np.asarray(y_event, dtype=int)
    X = np.asarray(X, dtype=float)

    n, p = X.shape

    # Auto-center intercept prior on log(median observed time)
    pri = cfg.priors
    mu_mean = float(pri.mu_mean)
    if pri.auto_center:
        med = float(np.median(y_time[y_time > 0]))
        mu_mean = float(np.log(max(med, cfg.min_positive_duration)))

    # Censoring arrays for pm.Censored
    # observed is min(event_time, censor_time); for right-censored, observed=censor and upper=censor
    observed = y_time.copy()
    upper = np.where(y_event == 1, np.inf, y_time)

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=mu_mean, sigma=pri.mu_sd)

        if p > 0:
            beta = pm.Normal("beta", mu=0.0, sigma=pri.beta_sd, shape=p)
            linpred = mu + pm.math.dot(X, beta)
        else:
            beta = None
            linpred = mu

        # sigma/kappa linkage (identifiable)
        if pri.use_kappa_prior:
            kappa = pm.Gamma("kappa", alpha=pri.kappa_gamma_shape, beta=pri.kappa_gamma_rate)
            sigma = pm.Deterministic("sigma", 1.0 / kappa)
        else:
            sigma = pm.Exponential("sigma", lam=pri.sigma_exponential_rate)
            kappa = pm.Deterministic("kappa", 1.0 / sigma)

        # Weibull scale = exp(linpred)
        scale = pm.Deterministic("scale", pm.math.exp(linpred))

        base = pm.Weibull.dist(alpha=kappa, beta=scale)

        # Right-censored likelihood
        y_obs = pm.Censored(
            "y_obs",
            base,
            lower=None,
            upper=upper,
            observed=observed,
        )

        idata = pm.sample(
            draws=cfg.mcmc.draws,
            tune=cfg.mcmc.tune,
            chains=cfg.mcmc.chains,
            target_accept=cfg.mcmc.target_accept,
            random_seed=cfg.mcmc.random_seed,
            progressbar=False,
        )

    # Tidy posterior draws
    post = az.extract(idata, group="posterior").to_pandas()

    # Rename beta columns for readability
    if p > 0 and "beta" in post.columns:
        # ArviZ may expand beta into beta[0], beta[1], ...
        # Normalize names to beta_0, beta_1, ...
        renamed = {}
        for c in post.columns:
            if c.startswith("beta["):
                idx = c.split("[", 1)[1].split("]", 1)[0]
                renamed[c] = f"beta_{idx}"
        post = post.rename(columns=renamed)

    # Keep only parameters needed for simulation
    keep_cols = [c for c in post.columns if c in {"mu", "sigma", "kappa"} or c.startswith("beta_")]
    return post[keep_cols].reset_index(drop=True)


# ============================================================
# Posterior predictive simulation + capacity scaling
# ============================================================

def posterior_predictive_times(
    *,
    posterior_draws: pd.DataFrame,
    X_pred: np.ndarray,
    n_times: int,
    cfg: SimulationConfig,
) -> np.ndarray:
    """
    Draws posterior predictive survival times as a mixture over posterior samples.

    Strategy:
      - pick posterior rows at random
      - for each selected row, compute scale = exp(mu + X_pred @ beta)
      - sample T ~ Weibull(kappa, scale)
    """
    rng = np.random.default_rng(cfg.mcmc.random_seed + 123)

    post = posterior_draws
    if len(post) == 0:
        raise ValueError("posterior_draws is empty")

    # Determine how many mixture components we use
    idx = rng.integers(0, len(post), size=n_times)

    # Parse betas
    beta_cols = sorted([c for c in post.columns if c.startswith("beta_")], key=lambda s: int(s.split("_")[1]))
    p = len(beta_cols)

    if p > 0:
        if X_pred.shape != (p,):
            raise ValueError(f"X_pred must have shape ({p},) to match {p} beta coefficients. Got {X_pred.shape}.")
        betas = post.loc[idx, beta_cols].to_numpy(dtype=float)
        lin = post.loc[idx, "mu"].to_numpy(dtype=float) + (betas @ X_pred.reshape(-1, 1)).reshape(-1)
    else:
        lin = post.loc[idx, "mu"].to_numpy(dtype=float)

    kappa = post.loc[idx, "kappa"].to_numpy(dtype=float)
    scale = np.exp(lin)

    # Sample Weibull: numpy uses shape a and scale
    # np.random.weibull(a) gives samples of Weibull(shape=a) with scale=1, so multiply by scale.
    t = rng.weibull(kappa) * scale

    # Enforce positive
    t = np.maximum(t, cfg.min_positive_duration)
    return t.astype(float)


def summarize_simulations(
    sim_baseline: np.ndarray,
    sim_capacity: np.ndarray,
    *,
    cfg: SimulationConfig,
    capacity_multiplier: float,
    transition_key: str,
    n_cases_used: int,
) -> pd.DataFrame:
    """
    Returns a 1-row DataFrame with summary stats for baseline and capacity scenario.
    """
    def _summ(x: np.ndarray, prefix: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_median": float(np.median(x)),
        }
        for q in cfg.summary_quantiles:
            out[f"{prefix}_q{int(round(q*100)):02d}"] = float(np.quantile(x, q))
        for tau in cfg.sla_thresholds_minutes:
            out[f"{prefix}_p_le_{int(tau)}m"] = float(np.mean(x <= tau))
        return out

    row: Dict[str, Any] = {
        "transition_key": transition_key,
        "capacity_multiplier": float(capacity_multiplier),
        "n_cases_used": int(n_cases_used),
        "n_mc_times": int(len(sim_baseline)),
    }
    row.update(_summ(sim_baseline, "baseline"))
    row.update(_summ(sim_capacity, "capacity"))

    return pd.DataFrame([row])


# ============================================================
# Internal helpers
# ============================================================

def _get_transition_spec(key: str, transitions: Sequence[TransitionSpec]) -> TransitionSpec:
    for t in transitions:
        if t.key == key:
            return t
    raise KeyError(
        f"Unknown transition_key {key!r}. "
        f"Valid keys: {[t.key for t in transitions]}"
    )


def _prep_df(event_log_df: pd.DataFrame, cfg: SimulationConfig) -> pd.DataFrame:
    df = event_log_df.copy()

    # Parse timestamps
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col])

    if cfg.normalize_activities:
        df[cfg.act_col] = (
            df[cfg.act_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    if cfg.sort_events:
        df = df.sort_values([cfg.case_col, cfg.time_col], kind="mergesort")

    return df


def _make_X_pred(cfg: SimulationConfig, covariate_values: Optional[Dict[str, Any]]) -> np.ndarray:
    """
    For now, covariates are optional and assumed numeric if provided.
    (You can extend this later with encoding logic once you choose covariates.)
    """
    cov_cols = list(cfg.covariate_cols) if cfg.covariate_cols else []
    if not cov_cols:
        return np.zeros((0,), dtype=float)

    if covariate_values is None:
        # default to zeros (interpretable as centered covariates)
        return np.zeros((len(cov_cols),), dtype=float)

    x = []
    for c in cov_cols:
        v = covariate_values.get(c, 0.0)
        x.append(float(v))
    return np.asarray(x, dtype=float)

### Notes
# This file assumes your standardized log has case_id, activity, timestamp, and that activity
# contains canonical labels like "registration", "assessment", "consult request", "triage", "discharge".
# Your Streamlit mapping UI can enforce that later.
#
# Capacity is applied exactly as you requested: T_capacity = T_baseline / m.
#
# You’ll need pymc + arviz installed to run MCMC:
# pip install pymc arviz
#
# If you later want covariates (age, triage_code, queue length at arrival), you can pass covariate_cols=[...]
# in SimulationConfig and provide covariate_values={...} for scenario predictions.