from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

TargetType = Literal["binary", "regression"]
CutoffType = Literal["minutes_since_first_event", "n_events"]

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
    X: pd.DataFrame                 # features (one row per case)
    y: Optional[pd.Series]          # target (None if not provided)
    case_index: pd.Index            # case ids aligned to X
    feature_info: Dict[str, Any]    # useful metadata for UI (top activities, etc.)
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
    per_case: pd.DataFrame  # case_id, y_true, y_pred, y_score (if available)
    task: TargetType


# ----------------------------
# Public: Build prediction dataset at a cutoff time
# ----------------------------

def build_prediction_dataset(
    event_log_df: pd.DataFrame,
    *,
    target_df: Optional[pd.DataFrame] = None,
    target_case_col: str = "case_id",
    target_col: Optional[str] = None,
    cfg: Optional[PredictionConfig] = None,
) -> DatasetBundle:
    """
    Build a case-level feature table for predictive modeling from an event log.

    event_log_df: long-format events, one row per event with case_id, timestamp, activity (and optional resource, covariates).
    target_df: case-level outcomes (one row per case). Optional; if provided we will join y.
    target_col: name of the target column in target_df. If None, y will be None.

    Returns X (features) and y (optional) aligned by case id.
    """
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

    # Determine "cutoff timestamp" per case (or n_events)
    sliced = _slice_events_at_cutoff(df, cfg=cfg)

    # Identify top-K activities from sliced events (for stable feature space)
    top_acts = (
        sliced[cfg.act_col]
        .value_counts()
        .head(cfg.top_k_activities)
        .index
        .tolist()
    )

    # Base features
    feats = _build_case_features(
        sliced,
        cfg=cfg,
        top_acts=top_acts,
    )

    # Join static covariates (if present in event_log_df; assumed constant per case)
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

    # Join target
    y = None
    if target_df is not None and target_col is not None:
        if target_case_col not in target_df.columns:
            raise KeyError(f"target_df missing target_case_col={target_case_col}. Found: {list(target_df.columns)}")
        if target_col not in target_df.columns:
            raise KeyError(f"target_df missing target_col={target_col}. Found: {list(target_df.columns)}")

        t = target_df[[target_case_col, target_col]].copy()
        t = t.rename(columns={target_case_col: cfg.case_col})
        feats = feats.merge(t, on=cfg.case_col, how="inner")  # only keep cases with label
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
    """
    Return subset of events up to a cutoff.
    - minutes_since_first_event: keep events with timestamp <= first_time + cutoff_minutes
    - n_events: keep first N events per case
    """
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
    """
    Builds case-level features:
      - n_events
      - elapsed_minutes (first -> last in slice)
      - time-of-day/day-of-week for first event (optional)
      - counts of top activities (optional)
      - last activity indicator (optional)
      - resource counts (optional)
    """
    case = cfg.case_col
    t = cfg.time_col
    a = cfg.act_col

    g = sliced.groupby(case, sort=False)

    out = pd.DataFrame({case: g.size().index})
    out["n_events"] = g.size().values.astype(int)

    # Elapsed time in slice
    tmin = g[t].min()
    tmax = g[t].max()
    out["elapsed_minutes"] = ((tmax - tmin).dt.total_seconds() / 60.0).values

    if cfg.include_time_features:
        # time features from first event timestamp
        first = tmin
        out["hour"] = first.dt.hour.values.astype(int)
        out["dayofweek"] = first.dt.dayofweek.values.astype(int)  # Monday=0

    if cfg.include_activity_counts:
        # activity counts for top acts
        # build in a stable set of columns: act_count::<act>
        sub = sliced[sliced[a].isin(top_acts)].copy()
        counts = (
            sub.groupby([case, a], sort=False)
               .size()
               .unstack(fill_value=0)
               .reindex(columns=top_acts, fill_value=0)
        )
        counts.columns = [f"act_count::{c}" for c in counts.columns]
        out = out.merge(counts.reset_index(), on=case, how="left")
        # fill missing (cases with none of top_acts)
        for c in counts.columns:
            if c not in out.columns:
                out[c] = 0
        out[counts.columns] = out[counts.columns].fillna(0).astype(int)

    if cfg.include_last_activity:
        last_act = g[a].last().astype(str)
        # One-hot last activity (restricted to top_acts + "OTHER")
        la = last_act.where(last_act.isin(top_acts), other="OTHER")
        d = pd.get_dummies(la, prefix="last_act", dtype=int)
        out = out.merge(d.reset_index().rename(columns={case: case}), left_on=case, right_on=case, how="left")

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


# ----------------------------
# Modeling: train / predict / evaluate
# ----------------------------

def train_model(
    dataset: DatasetBundle,
    *,
    task: TargetType,
    model_type: Literal["logreg", "rf", "xgb"] = "logreg",
    random_state: int = 42,
) -> ModelBundle:
    """
    Trains a basic regression model
    - binary: logistic regression (default) or random forest
    - regression: random forest regressor (default) or linear ridge

    Keeps everything as a ModelBundle so Streamlit can hold it in session state later.
    """
    if dataset.y is None:
        raise ValueError("dataset.y is None. Provide target_df + target_col to build_prediction_dataset().")

    X = dataset.X.copy()
    y = dataset.y.copy()

    # Basic preprocessing: numeric conversion + simple missing handling
    X = _coerce_numeric_frame(X)
    X = X.fillna(0)

    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

    # We'll return the fitted model; splitting/eval handled in evaluate_model
    if task == "binary":
        model = _make_binary_model(model_type=model_type, random_state=random_state)
    elif task == "regression":
        model = _make_regression_model(model_type=model_type, random_state=random_state)
    else:
        raise ValueError(f"Unknown task: {task}")

    model.fit(X, y)

    return ModelBundle(model=model, task=task, cfg=dataset.cfg, feature_columns=list(X.columns))


def predict(
    model_bundle: ModelBundle,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates predictions for a case-level feature frame X.
    Returns a DataFrame with columns:
      - y_pred
      - y_score (for binary, probability if available)
    """
    Xc = X.copy()
    Xc = _coerce_numeric_frame(Xc)
    Xc = Xc.fillna(0)

    # Align feature columns (important for Streamlit re-runs)
    for col in model_bundle.feature_columns:
        if col not in Xc.columns:
            Xc[col] = 0
    Xc = Xc[model_bundle.feature_columns]

    y_pred = model_bundle.model.predict(Xc)

    out = pd.DataFrame(index=Xc.index)
    out["y_pred"] = y_pred

    if model_bundle.task == "binary":
        # Prefer predict_proba if available
        if hasattr(model_bundle.model, "predict_proba"):
            out["y_score"] = model_bundle.model.predict_proba(Xc)[:, 1]
        elif hasattr(model_bundle.model, "decision_function"):
            # Map decision scores to (0,1) via sigmoid for UI display
            s = model_bundle.model.decision_function(Xc)
            out["y_score"] = 1 / (1 + np.exp(-s))
        else:
            out["y_score"] = np.nan

    return out.reset_index().rename(columns={"index": model_bundle.cfg.case_col})


def evaluate_model(
    dataset: DatasetBundle,
    *,
    task: TargetType,
    model_type: Literal["logreg", "rf", "xgb"] = "logreg",
    test_size: float = 0.25,
    random_state: int = 42,
) -> EvaluationResult:
    """
    Quick train/test evaluation (UI-friendly baseline).

    Returns:
      - metrics dict
      - per_case predictions for the test split
    """
    if dataset.y is None:
        raise ValueError("dataset.y is None. Provide target_df + target_col to build_prediction_dataset().")

    X = dataset.X.copy()
    y = dataset.y.copy()

    X = _coerce_numeric_frame(X).fillna(0)

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            roc_auc_score, average_precision_score,
            accuracy_score, f1_score,
            mean_absolute_error, mean_squared_error, r2_score
        )
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task == "binary" else None
    )

    if task == "binary":
        model = _make_binary_model(model_type=model_type, random_state=random_state)
        model.fit(X_train, y_train)

        y_hat = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            s = model.decision_function(X_test)
            y_score = 1 / (1 + np.exp(-s))
        else:
            y_score = None

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_hat)),
            "f1": float(f1_score(y_test, y_hat)),
        }
        if y_score is not None:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
            metrics["avg_precision"] = float(average_precision_score(y_test, y_score))

        per_case = pd.DataFrame({
            dataset.cfg.case_col: X_test.index,
            "y_true": y_test.values,
            "y_pred": y_hat,
            "y_score": y_score if y_score is not None else np.nan,
        })

        return EvaluationResult(metrics=metrics, per_case=per_case.reset_index(drop=True), task=task)

    elif task == "regression":
        model = _make_regression_model(model_type=model_type, random_state=random_state)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        metrics = {
            "mae": float(mean_absolute_error(y_test, y_hat)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_hat))),
            "r2": float(r2_score(y_test, y_hat)),
        }
        per_case = pd.DataFrame({
            dataset.cfg.case_col: X_test.index,
            "y_true": y_test.values,
            "y_pred": y_hat,
        })
        return EvaluationResult(metrics=metrics, per_case=per_case.reset_index(drop=True), task=task)

    else:
        raise ValueError(f"Unknown task: {task}")


# ----------------------------
# Models (simple baselines)
# ----------------------------

def _make_binary_model(model_type: str, random_state: int):
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as e:
        raise ImportError("scikit-learn is required. Install with: pip install scikit-learn") from e

    if model_type == "logreg":
        # Scaling helps logreg; keep it in a pipeline
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
        ])

    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

    # Placeholder: xgb reserved if you add xgboost later
    if model_type == "xgb":
        raise ValueError("model_type='xgb' selected but xgboost is not wired in. Use 'logreg' or 'rf' for now.")

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
        # In regression mode, treat "logreg" as ridge baseline
        return Ridge(alpha=1.0, random_state=random_state)

    if model_type == "xgb":
        raise ValueError("model_type='xgb' selected but xgboost is not wired in. Use 'rf' or 'logreg'(ridge).")

    raise ValueError(f"Unknown model_type: {model_type}")


# ----------------------------
# Utils
# ----------------------------

def _coerce_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert bool/int/float strings -> numeric where possible.
    Non-numeric columns are one-hot encoded (small cardinality) or dropped if too wide.
    This keeps dependencies minimal and is safe for Streamlit reruns.
    """
    Xc = X.copy()

    # Separate numeric-like columns
    numeric_cols = []
    object_cols = []
    for c in Xc.columns:
        if pd.api.types.is_numeric_dtype(Xc[c]):
            numeric_cols.append(c)
        else:
            object_cols.append(c)

    # Try coercing object cols to numeric
    for c in list(object_cols):
        coerced = pd.to_numeric(Xc[c], errors="coerce")
        # if most values become numeric, keep numeric
        if coerced.notna().mean() > 0.9:
            Xc[c] = coerced
            numeric_cols.append(c)
            object_cols.remove(c)

    # One-hot encode remaining object cols with limited cardinality
    keep_obj = []
    for c in object_cols:
        nunique = Xc[c].nunique(dropna=True)
        if nunique <= 50:
            keep_obj.append(c)

    if keep_obj:
        dummies = pd.get_dummies(Xc[keep_obj].fillna("NA"), prefix=keep_obj, dtype=int)
        Xc = Xc.drop(columns=keep_obj)
        Xc = pd.concat([Xc, dummies], axis=1)

    # Drop any remaining non-numeric columns (high-cardinality text)
    for c in Xc.columns:
        if not pd.api.types.is_numeric_dtype(Xc[c]):
            Xc = Xc.drop(columns=[c])

    return Xc
