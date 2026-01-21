from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TargetType = Literal["binary", "regression"]
CutoffType = Literal["minutes_since_first_event", "n_events"]

# ============================
# Core data structures
# ============================

@dataclass(frozen=True)
class PredictionConfig:
    # Event log schema
    case_col: str = "case_id"
    time_col: str = "timestamp"
    act_col: str = "activity"

    resource_col: Optional[str] = "resource"
    static_feature_cols: Optional[Sequence[str]] = None  # e.g., ["age", "triage_desc"]

    # How to slice the trace for "prediction time"
    cutoff_type: CutoffType = "minutes_since_first_event"
    cutoff_value: int = 30  # minutes or n_events depending on cutoff_type

    # Feature building
    top_k_activities: int = 50
    include_time_features: bool = True
    include_activity_counts: bool = True
    include_last_activity: bool = True
    include_resource_counts: bool = False  # set True if resource_col is meaningful as covariate

    # Cleaning
    sort_events: bool = True
    dropna_timestamps: bool = True


@dataclass(frozen=True)
class DatasetBundle:
    X: pd.DataFrame
    y: Optional[pd.Series]
    case_index: pd.Index
    feature_info: Dict[str, Any]
    cfg: PredictionConfig


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    task: TargetType
    cfg: PredictionConfig
    feature_columns: List[str]


@dataclass(frozen=True)
class EvaluationResult:
    metrics: Dict[str, float]
    per_case: pd.DataFrame
    task: TargetType


# ============================
# Targets: required 3 outcomes
# ============================

TargetName = Literal["prob_admission", "prob_lwbs", "remaining_time_to_pia"]


@dataclass(frozen=True)
class TargetSpec:
    name: TargetName
    task: TargetType
    target_col: str


@dataclass(frozen=True)
class TargetBuildConfig:
    # Event log schema
    case_col: str = "case_id"
    time_col: str = "timestamp"
    act_col: str = "activity"

    # Case outcome columns
    disposition_col: Optional[str] = "disposition_desc"

    # Keywords (case-insensitive substring match)
    admission_keywords: Tuple[str, ...] = ("admit", "admission", "inpatient")
    lwbs_keywords: Tuple[str, ...] = ("lwbs", "left without being seen", "left after triage")

    # PIA event definition (picked in UI)
    pia_activity: str = "PIA"

    # If provided, arrival time uses first occurrence of this activity
    arrival_activity: Optional[str] = None

    # Data quality
    require_pia: bool = True
    cap_remaining_minutes: Optional[float] = 24 * 60.0


def _normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def _first_time_per_case(df: pd.DataFrame, case_col: str, time_col: str) -> pd.Series:
    return df.groupby(case_col, sort=False)[time_col].min()


def _arrival_time_per_case(df: pd.DataFrame, cfg: TargetBuildConfig) -> pd.Series:
    if cfg.arrival_activity is None:
        return _first_time_per_case(df, cfg.case_col, cfg.time_col)

    mask = df[cfg.act_col].astype(str) == str(cfg.arrival_activity)
    arr = df.loc[mask].groupby(cfg.case_col, sort=False)[cfg.time_col].min()
    first = _first_time_per_case(df, cfg.case_col, cfg.time_col)
    return arr.reindex(first.index).fillna(first)


def _cutoff_time_per_case(df: pd.DataFrame, pred_cfg: PredictionConfig) -> pd.Series:
    if pred_cfg.cutoff_type == "minutes_since_first_event":
        first = df.groupby(pred_cfg.case_col, sort=False)[pred_cfg.time_col].transform("min")
        cutoff_ts = first + pd.to_timedelta(pred_cfg.cutoff_value, unit="m")
        return cutoff_ts.groupby(df[pred_cfg.case_col], sort=False).first()

    if pred_cfg.cutoff_type == "n_events":
        g = df.sort_values([pred_cfg.case_col, pred_cfg.time_col], kind="mergesort").groupby(
            pred_cfg.case_col, sort=False
        )
        nth = g.nth(pred_cfg.cutoff_value - 1)[pred_cfg.time_col]
        last = g[pred_cfg.time_col].max()
        return nth.reindex(last.index).fillna(last)

    raise ValueError(f"Unknown cutoff_type: {pred_cfg.cutoff_type}")


def build_target_prob_admission(
    df_events: pd.DataFrame,
    *,
    cfg: TargetBuildConfig,
    target_col: str = "y_admission",
) -> pd.DataFrame:
    if not cfg.disposition_col or cfg.disposition_col not in df_events.columns:
        raise KeyError(
            f"To build admission target, disposition_col={cfg.disposition_col!r} must exist."
        )

    disp = (
        df_events[[cfg.case_col, cfg.disposition_col]]
        .dropna(subset=[cfg.disposition_col])
        .groupby(cfg.case_col, sort=False)[cfg.disposition_col]
        .first()
    )
    s = _normalize_text_series(disp)
    kw = tuple(k.lower() for k in cfg.admission_keywords)
    y = s.apply(lambda x: any(k in x for k in kw)).astype(int)

    return pd.DataFrame({cfg.case_col: y.index, target_col: y.values}).reset_index(drop=True)


def build_target_prob_lwbs(
    df_events: pd.DataFrame,
    *,
    cfg: TargetBuildConfig,
    target_col: str = "y_lwbs",
) -> pd.DataFrame:
    if not cfg.disposition_col or cfg.disposition_col not in df_events.columns:
        raise KeyError(
            f"To build LWBS target, disposition_col={cfg.disposition_col!r} must exist."
        )

    disp = (
        df_events[[cfg.case_col, cfg.disposition_col]]
        .dropna(subset=[cfg.disposition_col])
        .groupby(cfg.case_col, sort=False)[cfg.disposition_col]
        .first()
    )
    s = _normalize_text_series(disp)
    kw = tuple(k.lower() for k in cfg.lwbs_keywords)
    y = s.apply(lambda x: any(k in x for k in kw)).astype(int)

    return pd.DataFrame({cfg.case_col: y.index, target_col: y.values}).reset_index(drop=True)


def build_target_remaining_time_to_pia(
    df_events: pd.DataFrame,
    *,
    cfg: TargetBuildConfig,
    pred_cfg: PredictionConfig,
    target_col: str = "y_remaining_to_pia_min",
) -> pd.DataFrame:
    """
    Remaining time from feature cutoff to first PIA event:
      remaining = max(0, pia_time - cutoff_time)
    """
    df = df_events.copy()
    df[pred_cfg.time_col] = pd.to_datetime(df[pred_cfg.time_col], errors="coerce", utc=True)
    df = df.dropna(subset=[pred_cfg.time_col]).copy()
    df[pred_cfg.act_col] = df[pred_cfg.act_col].astype(str)

    cutoff_ts = _cutoff_time_per_case(df, pred_cfg)

    pia_mask = df[pred_cfg.act_col] == str(cfg.pia_activity)
    pia_ts = df.loc[pia_mask].groupby(pred_cfg.case_col, sort=False)[pred_cfg.time_col].min()

    cases = cutoff_ts.index
    out = pd.DataFrame({pred_cfg.case_col: cases})
    out["cutoff_time"] = cutoff_ts.reindex(cases).values
    out["pia_time"] = pia_ts.reindex(cases).values
    out[target_col] = ((out["pia_time"] - out["cutoff_time"]).dt.total_seconds() / 60.0).astype(float)

    if cfg.require_pia:
        out = out.dropna(subset=[target_col]).copy()

    out = out[out[target_col].notna()].copy()
    out = out[out[target_col] >= 0].copy()

    if cfg.cap_remaining_minutes is not None:
        out = out[out[target_col] <= float(cfg.cap_remaining_minutes)].copy()

    return out[[pred_cfg.case_col, target_col]].reset_index(drop=True)


def build_required_target(
    event_log_df: pd.DataFrame,
    *,
    which: TargetName,
    tcfg: TargetBuildConfig,
    pred_cfg: PredictionConfig,
) -> Tuple[pd.DataFrame, TargetSpec]:
    if which == "prob_admission":
        spec = TargetSpec(name=which, task="binary", target_col="y_admission")
        return build_target_prob_admission(event_log_df, cfg=tcfg, target_col=spec.target_col), spec

    if which == "prob_lwbs":
        spec = TargetSpec(name=which, task="binary", target_col="y_lwbs")
        return build_target_prob_lwbs(event_log_df, cfg=tcfg, target_col=spec.target_col), spec

    if which == "remaining_time_to_pia":
        spec = TargetSpec(name=which, task="regression", target_col="y_remaining_to_pia_min")
        return (
            build_target_remaining_time_to_pia(event_log_df, cfg=tcfg, pred_cfg=pred_cfg, target_col=spec.target_col),
            spec,
        )

    raise ValueError(f"Unknown target: {which}")


# ============================
# Build prediction dataset (features + optional target join)
# ============================

def build_prediction_dataset(
    event_log_df: pd.DataFrame,
    *,
    target_df: Optional[pd.DataFrame] = None,
    target_case_col: str = "case_id",
    target_col: Optional[str] = None,
    cfg: Optional[PredictionConfig] = None,
) -> DatasetBundle:
    if cfg is None:
        cfg = PredictionConfig()

    required = {cfg.case_col, cfg.time_col, cfg.act_col}
    missing = required - set(event_log_df.columns)
    if missing:
        raise KeyError(f"event_log_df missing required columns: {missing}. Found: {list(event_log_df.columns)}")

    df = event_log_df.copy()

    if cfg.dropna_timestamps:
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
        df = df.dropna(subset=[cfg.time_col])

    if cfg.sort_events:
        df = df.sort_values([cfg.case_col, cfg.time_col], kind="mergesort")

    df[cfg.act_col] = df[cfg.act_col].astype(str)
    if cfg.resource_col and cfg.resource_col in df.columns:
        df[cfg.resource_col] = df[cfg.resource_col].astype(str)

    sliced = _slice_events_at_cutoff(df, cfg=cfg)

    top_acts = sliced[cfg.act_col].value_counts().head(cfg.top_k_activities).index.tolist()

    feats = _build_case_features(sliced, cfg=cfg, top_acts=top_acts)

    if cfg.static_feature_cols:
        static_cols = [c for c in cfg.static_feature_cols if c in df.columns]
        if static_cols:
            stat = (
                df[[cfg.case_col] + static_cols]
                .groupby(cfg.case_col, sort=False)
                .first()
                .reset_index()
            )
            feats = feats.merge(stat, on=cfg.case_col, how="left")

    y = None
    if target_df is not None and target_col is not None:
        if target_case_col not in target_df.columns:
            raise KeyError(f"target_df missing target_case_col={target_case_col}. Found: {list(target_df.columns)}")
        if target_col not in target_df.columns:
            raise KeyError(f"target_df missing target_col={target_col}. Found: {list(target_df.columns)}")

        t = target_df[[target_case_col, target_col]].copy().rename(columns={target_case_col: cfg.case_col})
        feats = feats.merge(t, on=cfg.case_col, how="inner")
        y = feats[target_col]
        X = feats.drop(columns=[target_col])
    else:
        X = feats

    X = X.set_index(cfg.case_col)
    case_index = X.index.copy()

    feature_info = {
        "top_activities": top_acts,
        "cutoff_type": cfg.cutoff_type,
        "cutoff_value": cfg.cutoff_value,
    }

    return DatasetBundle(X=X, y=y, case_index=case_index, feature_info=feature_info, cfg=cfg)


def _slice_events_at_cutoff(df: pd.DataFrame, *, cfg: PredictionConfig) -> pd.DataFrame:
    if cfg.cutoff_type == "minutes_since_first_event":
        first_time = df.groupby(cfg.case_col, sort=False)[cfg.time_col].transform("min")
        cutoff_ts = first_time + pd.to_timedelta(cfg.cutoff_value, unit="m")
        return df[df[cfg.time_col] <= cutoff_ts].copy()

    if cfg.cutoff_type == "n_events":
        return df.groupby(cfg.case_col, sort=False).head(cfg.cutoff_value).copy()

    raise ValueError(f"Unknown cutoff_type: {cfg.cutoff_type}")


def _build_case_features(
    sliced: pd.DataFrame,
    *,
    cfg: PredictionConfig,
    top_acts: List[str],
) -> pd.DataFrame:
    case = cfg.case_col
    t = cfg.time_col
    a = cfg.act_col

    g = sliced.groupby(case, sort=False)

    out = pd.DataFrame({case: g.size().index})
    out["n_events"] = g.size().values.astype(int)

    tmin = g[t].min()
    tmax = g[t].max()
    out["elapsed_minutes"] = ((tmax - tmin).dt.total_seconds() / 60.0).values

    if cfg.include_time_features:
        first = tmin
        out["hour"] = first.dt.hour.values.astype(int)
        out["dayofweek"] = first.dt.dayofweek.values.astype(int)

    if cfg.include_activity_counts:
        sub = sliced[sliced[a].isin(top_acts)].copy()
        counts = (
            sub.groupby([case, a], sort=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=top_acts, fill_value=0)
        )
        counts.columns = [f"act_count::{c}" for c in counts.columns]
        out = out.merge(counts.reset_index(), on=case, how="left")
        out[counts.columns] = out[counts.columns].fillna(0).astype(int)

    if cfg.include_last_activity:
        last_act = g[a].last().astype(str)
        la = last_act.where(last_act.isin(top_acts), other="OTHER")
        d = pd.get_dummies(la, prefix="last_act", dtype=int)
        out = out.merge(d.reset_index(), on=case, how="left")

    if cfg.include_resource_counts and cfg.resource_col and cfg.resource_col in sliced.columns:
        r = cfg.resource_col
        top_r = sliced[r].value_counts().head(25).index.tolist()
        subr = sliced[sliced[r].isin(top_r)].copy()
        rc = (
            subr.groupby([case, r], sort=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=top_r, fill_value=0)
        )
        rc.columns = [f"res_count::{c}" for c in rc.columns]
        out = out.merge(rc.reset_index(), on=case, how="left")
        out[rc.columns] = out[rc.columns].fillna(0).astype(int)

    return out


# ============================
# Modeling: train / predict / evaluate
# ============================

def train_model(
    dataset: DatasetBundle,
    *,
    task: TargetType,
    model_type: Literal["logreg", "rf"] = "logreg",
    random_state: int = 42,
) -> ModelBundle:
    if dataset.y is None:
        raise ValueError("dataset.y is None. Provide target_df + target_col to build_prediction_dataset().")

    X = _coerce_numeric_frame(dataset.X.copy()).fillna(0)
    y = dataset.y.copy()

    if task == "binary":
        model = _make_binary_model(model_type=model_type, random_state=random_state)
    elif task == "regression":
        model = _make_regression_model(model_type=model_type, random_state=random_state)
    else:
        raise ValueError(f"Unknown task: {task}")

    model.fit(X, y)

    return ModelBundle(model=model, task=task, cfg=dataset.cfg, feature_columns=list(X.columns))


def predict(model_bundle: ModelBundle, X: pd.DataFrame) -> pd.DataFrame:
    Xc = _coerce_numeric_frame(X.copy()).fillna(0)

    for col in model_bundle.feature_columns:
        if col not in Xc.columns:
            Xc[col] = 0
    Xc = Xc[model_bundle.feature_columns]

    y_pred = model_bundle.model.predict(Xc)

    out = pd.DataFrame(index=Xc.index)
    out["y_pred"] = y_pred

    if model_bundle.task == "binary":
        if hasattr(model_bundle.model, "predict_proba"):
            out["y_score"] = model_bundle.model.predict_proba(Xc)[:, 1]
        elif hasattr(model_bundle.model, "decision_function"):
            s = model_bundle.model.decision_function(Xc)
            out["y_score"] = 1 / (1 + np.exp(-s))
        else:
            out["y_score"] = np.nan

    return out.reset_index().rename(columns={"index": model_bundle.cfg.case_col})


def evaluate_model(
    dataset: DatasetBundle,
    *,
    task: TargetType,
    model_type: Literal["logreg", "rf"] = "logreg",
    test_size: float = 0.25,
    random_state: int = 42,
) -> EvaluationResult:
    if dataset.y is None:
        raise ValueError("dataset.y is None. Provide target_df + target_col to build_prediction_dataset().")

    X = _coerce_numeric_frame(dataset.X.copy()).fillna(0)
    y = dataset.y.copy()

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            balanced_accuracy_score,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
        )
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task == "binary" else None,
    )

    if task == "binary":
        model = _make_binary_model(model_type=model_type, random_state=random_state)
        model.fit(X_train, y_train)

        y_hat = model.predict(X_test)

        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            s = model.decision_function(X_test)
            y_score = 1 / (1 + np.exp(-s))

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_hat)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_hat)),
            "precision": float(precision_score(y_test, y_hat, zero_division=0)),
            "recall": float(recall_score(y_test, y_hat, zero_division=0)),
            "f1": float(f1_score(y_test, y_hat, zero_division=0)),
        }
        if y_score is not None and np.isfinite(y_score).all():
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
            metrics["avg_precision"] = float(average_precision_score(y_test, y_score))

        per_case = pd.DataFrame(
            {
                dataset.cfg.case_col: X_test.index,
                "y_true": y_test.values,
                "y_pred": y_hat,
                "y_score": y_score if y_score is not None else np.nan,
            }
        )
        return EvaluationResult(metrics=metrics, per_case=per_case.reset_index(drop=True), task=task)

    if task == "regression":
        model = _make_regression_model(model_type=model_type, random_state=random_state)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, y_hat)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_hat))),
            "r2": float(r2_score(y_test, y_hat)),
        }
        per_case = pd.DataFrame(
            {
                dataset.cfg.case_col: X_test.index,
                "y_true": y_test.values,
                "y_pred": y_hat,
            }
        )
        return EvaluationResult(metrics=metrics, per_case=per_case.reset_index(drop=True), task=task)

    raise ValueError(f"Unknown task: {task}")


def _make_binary_model(model_type: str, random_state: int):
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

    if model_type == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        )

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def _make_regression_model(model_type: str, random_state: int):
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_type == "logreg":
        return Ridge(alpha=1.0, random_state=random_state)

    raise ValueError(f"Unknown model_type: {model_type}")


# ============================
# CI helpers (historical + patient-level)
# ============================

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    No SciPy needed; uses a normal approx z=1.96 for alpha=0.05 by default.
    """
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    z = 1.959963984540054  # ~N(0,1) 97.5% quantile
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (float(lo), float(hi))


def bootstrap_ci(
    values: np.ndarray,
    stat: str = "mean",
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for mean/median/p90. Returns (estimate, lo, hi).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)

    if stat == "mean":
        est = float(np.mean(v))
        fn = np.mean
    elif stat == "median":
        est = float(np.median(v))
        fn = np.median
    elif stat == "p90":
        est = float(np.quantile(v, 0.90))
        fn = lambda x: np.quantile(x, 0.90)
    else:
        raise ValueError("stat must be one of: mean, median, p90")

    rng = np.random.default_rng(int(random_state))
    boots = []
    for _ in range(int(n_boot)):
        samp = rng.choice(v, size=v.size, replace=True)
        boots.append(float(fn(samp)))

    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (est, lo, hi)


def summarize_target_with_ci(
    target_df: pd.DataFrame,
    *,
    case_col: str,
    target_col: str,
    task: TargetType,
    alpha: float = 0.05,
    n_boot: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    UI-friendly summary + 95% CI:
      - binary: rate + Wilson CI
      - regression: mean/median/p90 + bootstrap CI (mean only is usually enough, but we provide all)
    """
    if target_df is None or target_df.empty:
        return pd.DataFrame(
            [
                {
                    "target": target_col,
                    "n_cases": 0,
                    "rate": np.nan,
                    "rate_ci_low": np.nan,
                    "rate_ci_high": np.nan,
                    "mean": np.nan,
                    "mean_ci_low": np.nan,
                    "mean_ci_high": np.nan,
                    "median": np.nan,
                    "median_ci_low": np.nan,
                    "median_ci_high": np.nan,
                    "p90": np.nan,
                    "p90_ci_low": np.nan,
                    "p90_ci_high": np.nan,
                }
            ]
        )

    n_cases = int(target_df[case_col].nunique())
    y = pd.to_numeric(target_df[target_col], errors="coerce").to_numpy()

    row: Dict[str, Any] = {"target": target_col, "n_cases": n_cases}

    if task == "binary":
        yb = y[np.isfinite(y)]
        k = int(np.sum(yb >= 0.5))  # y is 0/1, but be robust
        n = int(yb.size)
        rate = float(k / n) if n else np.nan
        lo, hi = wilson_ci(k, n, alpha=alpha)
        row.update(
            {
                "rate": rate,
                "rate_ci_low": lo,
                "rate_ci_high": hi,
                "mean": np.nan,
                "mean_ci_low": np.nan,
                "mean_ci_high": np.nan,
                "median": np.nan,
                "median_ci_low": np.nan,
                "median_ci_high": np.nan,
                "p90": np.nan,
                "p90_ci_low": np.nan,
                "p90_ci_high": np.nan,
            }
        )
        return pd.DataFrame([row])

    # regression
    est_m, lo_m, hi_m = bootstrap_ci(y, stat="mean", alpha=alpha, n_boot=n_boot, random_state=random_state)
    est_med, lo_med, hi_med = bootstrap_ci(y, stat="median", alpha=alpha, n_boot=n_boot, random_state=random_state)
    est_p90, lo_p90, hi_p90 = bootstrap_ci(y, stat="p90", alpha=alpha, n_boot=n_boot, random_state=random_state)

    row.update(
        {
            "rate": np.nan,
            "rate_ci_low": np.nan,
            "rate_ci_high": np.nan,
            "mean": est_m,
            "mean_ci_low": lo_m,
            "mean_ci_high": hi_m,
            "median": est_med,
            "median_ci_low": lo_med,
            "median_ci_high": hi_med,
            "p90": est_p90,
            "p90_ci_low": lo_p90,
            "p90_ci_high": hi_p90,
        }
    )
    return pd.DataFrame([row])


def bootstrap_prediction_ci(
    event_log_df: pd.DataFrame,
    *,
    pred_cfg: PredictionConfig,
    tcfg: TargetBuildConfig,
    which: TargetName,
    X_row: pd.DataFrame,
    model_type: Literal["logreg", "rf"] = "logreg",
    alpha: float = 0.05,
    n_boot: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Patient-level CI by bootstrap refit:
      - resample cases with replacement
      - rebuild dataset, fit, predict X_row
    Returns one-row DF with estimate + CI.
    This can be slow; gate behind a checkbox in Streamlit.
    """
    rng = np.random.default_rng(int(random_state))

    # Build full target + dataset once to get aligned case universe
    target_df, spec = build_required_target(event_log_df, which=which, tcfg=tcfg, pred_cfg=pred_cfg)
    dataset = build_prediction_dataset(
        event_log_df,
        target_df=target_df,
        target_case_col=pred_cfg.case_col,
        target_col=spec.target_col,
        cfg=pred_cfg,
    )

    cases = dataset.X.index.to_numpy()
    if cases.size < 50:
        raise ValueError("Too few cases to bootstrap patient-level CIs (need ~50+).")

    # Fit on full data for point estimate
    mb_full = train_model(dataset, task=spec.task, model_type=model_type, random_state=random_state)
    pred_full = predict(mb_full, X_row.set_index(pd.Index(["profile"])))
    if spec.task == "binary":
        point = float(pred_full["y_score"].iloc[0])
    else:
        point = float(pred_full["y_pred"].iloc[0])

    samples = []
    for b in range(int(n_boot)):
        samp_cases = rng.choice(cases, size=cases.size, replace=True)
        # Subsample X/y by cases (fast); refit
        Xb = dataset.X.loc[samp_cases]
        yb = dataset.y.loc[samp_cases] if dataset.y is not None else None
        db = DatasetBundle(
            X=Xb,
            y=yb,
            case_index=Xb.index,
            feature_info=dataset.feature_info,
            cfg=dataset.cfg,
        )
        mb = train_model(db, task=spec.task, model_type=model_type, random_state=int(random_state + b + 1))
        pb = predict(mb, X_row.set_index(pd.Index(["profile"])))
        samples.append(float(pb["y_score"].iloc[0] if spec.task == "binary" else pb["y_pred"].iloc[0]))

    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))

    out = {
        "target": which,
        "task": spec.task,
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": int(n_boot),
    }
    return pd.DataFrame([out])


# ============================
# Metric dropdown helper
# ============================

DEFAULT_METRIC_SETS: Dict[str, List[str]] = {
    "Binary (yes/no)": [
        "f1",
        "recall",
        "precision",
        "balanced_accuracy",
        "roc_auc",
        "avg_precision",
        "accuracy",
    ],
    "Regression (time)": [
        "mae",
        "rmse",
        "r2",
    ],
}


# ============================
# Utils
# ============================

def _coerce_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()

    object_cols = [c for c in Xc.columns if not pd.api.types.is_numeric_dtype(Xc[c])]

    # Try coercing objects to numeric
    for c in list(object_cols):
        coerced = pd.to_numeric(Xc[c], errors="coerce")
        if coerced.notna().mean() > 0.9:
            Xc[c] = coerced
            object_cols.remove(c)

    # One-hot encode remaining objects with limited cardinality
    keep_obj = []
    for c in object_cols:
        if Xc[c].nunique(dropna=True) <= 50:
            keep_obj.append(c)

    if keep_obj:
        dummies = pd.get_dummies(Xc[keep_obj].fillna("NA"), prefix=keep_obj, dtype=int)
        Xc = Xc.drop(columns=keep_obj)
        Xc = pd.concat([Xc, dummies], axis=1)

    # Drop any remaining non-numeric
    drop = [c for c in Xc.columns if not pd.api.types.is_numeric_dtype(Xc[c])]
    if drop:
        Xc = Xc.drop(columns=drop)

    return Xc
