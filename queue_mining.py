# queue_mining.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Any

import os
import pandas as pd

# Optional import: your canonicalization layer
# You said the file is called event_log_organizer.py
try:
    from event_log_organizer import (
        canonicalize_event_log,
        CanonicalizationConfig,
        CanonicalizationReport,
    )
except Exception:
    canonicalize_event_log = None  # type: ignore
    CanonicalizationConfig = Any   # type: ignore
    CanonicalizationReport = Any   # type: ignore


# -----------------------------
# Config + result containers
# -----------------------------
@dataclass(frozen=True)
class QueueMiningConfig:
    case_col: str = "case_id"
    time_col: str = "timestamp"

    # STATE = the column you want to treat as the "queue state"
    state_col: str = "activity"

    # Optional column to carry along for labeling/debugging (NOT used for queue counting)
    covariate_col: Optional[str] = None  # e.g., "initial_zone"

    sort_events: bool = True
    drop_missing_state: bool = True

    # Default OFF (safer).
    collapse_consecutive_same_state: bool = False

    # Interval end rule: how to define leaving a state
    end_rule: Literal["next_event_time", "case_end_time"] = "next_event_time"

    max_events_per_case: Optional[int] = None
    parse_utc: bool = True
    debug: bool = False


@dataclass(frozen=True)
class StateIntervalsResult:
    intervals: pd.DataFrame
    config: QueueMiningConfig


@dataclass(frozen=True)
class QueueLengthResult:
    change_points: pd.DataFrame
    sampled: Optional[pd.DataFrame]
    config: QueueMiningConfig


@dataclass(frozen=True)
class QueueMiningRunResult:
    """Full pipeline output: preprocessing report (optional) + intervals + queues + summaries."""
    preprocessed_df: pd.DataFrame
    preprocess_report: Optional["CanonicalizationReport"]
    intervals_result: StateIntervalsResult
    queue_result: QueueLengthResult
    wait_summary: pd.DataFrame


# -----------------------------
# Small file helpers
# -----------------------------
def ensure_output_dir(output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_csv(df: pd.DataFrame, filename: str, *, output_dir: str = "output files") -> str:
    ensure_output_dir(output_dir)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path


# -----------------------------
# Validation helpers
# -----------------------------
def _validate_event_log(df: pd.DataFrame, cfg: QueueMiningConfig) -> None:
    missing = [c for c in [cfg.case_col, cfg.time_col, cfg.state_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    if cfg.debug:
        print(f"[validate] rows={len(df):,} cols={len(df.columns)}")
        print(f"[validate] unique cases={df[cfg.case_col].nunique(dropna=True):,}")
        print(f"[validate] unique states(raw)={df[cfg.state_col].nunique(dropna=True):,}")


# -----------------------------
# Core: build intervals
# -----------------------------
def build_state_intervals(
    event_log_df: pd.DataFrame,
    *,
    cfg: Optional[QueueMiningConfig] = None,
) -> StateIntervalsResult:
    if cfg is None:
        cfg = QueueMiningConfig()

    _validate_event_log(event_log_df, cfg)

    df = event_log_df.copy()

    # Ensure datetime
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce", utc=cfg.parse_utc)
    before = len(df)
    df = df.dropna(subset=[cfg.time_col])
    if cfg.debug:
        print(f"[intervals] dropped invalid timestamps: {before - len(df):,}")

    # Drop missing state
    if cfg.drop_missing_state:
        before = len(df)
        df = df.dropna(subset=[cfg.state_col])
        if cfg.debug:
            print(f"[intervals] dropped missing state: {before - len(df):,}")

    df[cfg.state_col] = df[cfg.state_col].astype(str)

    # Optional covariate column (carry along, not used as state)
    if cfg.covariate_col and cfg.covariate_col in df.columns:
        df["_cov"] = df[cfg.covariate_col].astype(str)
    else:
        df["_cov"] = None

    # Sort events
    if cfg.sort_events:
        df = df.sort_values([cfg.case_col, cfg.time_col], kind="mergesort")

    # Optional cap
    if cfg.max_events_per_case is not None:
        df = df.groupby(cfg.case_col, sort=False).head(cfg.max_events_per_case).copy()

    # De-duplicate exact same event at same time for same case
    df = df.drop_duplicates(subset=[cfg.case_col, cfg.time_col, cfg.state_col], keep="last")

    # Next event time and next state
    df["_next_time"] = df.groupby(cfg.case_col, sort=False)[cfg.time_col].shift(-1)
    df["_next_state"] = df.groupby(cfg.case_col, sort=False)[cfg.state_col].shift(-1)

    if cfg.collapse_consecutive_same_state:
        keep = (df[cfg.state_col] != df["_next_state"]) | df["_next_state"].isna()
        df = df[keep].copy()

    # Define end time
    if cfg.end_rule == "next_event_time":
        df["_end_time"] = df["_next_time"]
    elif cfg.end_rule == "case_end_time":
        last_time = df.groupby(cfg.case_col, sort=False)[cfg.time_col].transform("max")
        df["_end_time"] = last_time
    else:
        raise ValueError(f"Unknown end_rule: {cfg.end_rule}")

    before = len(df)
    df = df.dropna(subset=["_end_time"])
    if cfg.debug:
        print(f"[intervals] dropped missing end_time: {before - len(df):,}")

    intervals = pd.DataFrame(
        {
            cfg.case_col: df[cfg.case_col].values,
            "state": df[cfg.state_col].values,
            "start_time": df[cfg.time_col].values,
            "end_time": df["_end_time"].values,
            "covariate_value": df["_cov"].values,
        }
    )

    dt = intervals["end_time"] - intervals["start_time"]
    intervals["duration_seconds"] = dt.dt.total_seconds().astype(float)

    before = len(intervals)
    intervals = intervals[intervals["duration_seconds"] > 0].copy()
    if cfg.debug:
        print(f"[intervals] removed non-positive durations: {before - len(intervals):,}")

    intervals["duration_minutes"] = intervals["duration_seconds"] / 60.0
    intervals["date"] = intervals["start_time"].dt.date.astype(str)

    if cfg.debug:
        print(f"[intervals] intervals={len(intervals):,} unique states={intervals['state'].nunique():,}")

    return StateIntervalsResult(intervals=intervals.reset_index(drop=True), config=cfg)


# -----------------------------
# Queue length: change points
# -----------------------------
def compute_queue_length_change_points(
    intervals_df: pd.DataFrame,
    *,
    state_col: str = "state",
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> pd.DataFrame:
    if intervals_df is None or intervals_df.empty:
        return pd.DataFrame(columns=["state", "timestamp", "delta", "queue_length"])

    starts = intervals_df[[state_col, start_col]].copy()
    starts = starts.rename(columns={start_col: "timestamp"})
    starts["delta"] = 1

    ends = intervals_df[[state_col, end_col]].copy()
    ends = ends.rename(columns={end_col: "timestamp"})
    ends["delta"] = -1

    changes = pd.concat([starts, ends], ignore_index=True)
    changes["timestamp"] = pd.to_datetime(changes["timestamp"], errors="coerce", utc=True)
    changes = changes.dropna(subset=["timestamp"])
    if changes.empty:
        return pd.DataFrame(columns=["state", "timestamp", "delta", "queue_length"])

    changes = changes.sort_values([state_col, "timestamp", "delta"], kind="mergesort")
    changes["queue_length"] = changes.groupby(state_col, sort=False)["delta"].cumsum()

    return changes.reset_index(drop=True)


# -----------------------------
# Queue length: sampled grid
# -----------------------------
def sample_queue_lengths(
    change_points_df: pd.DataFrame,
    *,
    freq: str = "15min",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    if change_points_df is None or change_points_df.empty:
        return pd.DataFrame(columns=["state", "timestamp", "queue_length"])

    cp = change_points_df.copy()
    cp["timestamp"] = pd.to_datetime(cp["timestamp"], errors="coerce", utc=True)
    cp = cp.dropna(subset=["timestamp"])
    if cp.empty:
        return pd.DataFrame(columns=["state", "timestamp", "queue_length"])

    if start is None:
        start = cp["timestamp"].min()
    if end is None:
        end = cp["timestamp"].max()

    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=["state", "timestamp", "queue_length"])

    grid = pd.date_range(start=start, end=end, freq=freq)

    out_rows: List[pd.DataFrame] = []
    cp = cp.sort_values(["state", "timestamp"], kind="mergesort")

    for state, g in cp.groupby("state", sort=False):
        g = g.sort_values("timestamp", kind="mergesort")
        g = g.groupby("timestamp", as_index=False)["queue_length"].last()
        g = g.sort_values("timestamp", kind="mergesort")

        df_grid = pd.DataFrame({"timestamp": grid}).sort_values("timestamp", kind="mergesort")

        merged = pd.merge_asof(
            df_grid,
            g,
            on="timestamp",
            direction="backward",
            allow_exact_matches=True,
        )

        merged["queue_length"] = merged["queue_length"].fillna(0).astype(float)
        merged["state"] = state
        out_rows.append(merged[["state", "timestamp", "queue_length"]])

    if not out_rows:
        return pd.DataFrame(columns=["state", "timestamp", "queue_length"])

    return pd.concat(out_rows, ignore_index=True)


def compute_queue_lengths(
    intervals_result: StateIntervalsResult,
    *,
    sample_freq: Optional[str] = "15min",
) -> QueueLengthResult:
    cp = compute_queue_length_change_points(intervals_result.intervals)

    sampled = None
    if sample_freq is not None:
        sampled = sample_queue_lengths(cp, freq=sample_freq)
        if sampled.empty:
            sampled = None

    return QueueLengthResult(change_points=cp, sampled=sampled, config=intervals_result.config)


# -----------------------------
# Waiting time summaries
# -----------------------------
def summarize_wait_times(
    intervals_df: pd.DataFrame,
    *,
    by: str = "state",
    duration_col: str = "duration_minutes",
) -> pd.DataFrame:
    if intervals_df is None or intervals_df.empty:
        return pd.DataFrame(
            columns=[by, "n_intervals", "mean_min", "median_min", "p75_min", "p90_min", "p95_min", "max_min"]
        )

    g = intervals_df.groupby(by, dropna=False)[duration_col]
    summary = pd.DataFrame(
        {
            "n_intervals": g.size(),
            "mean_min": g.mean(),
            "median_min": g.median(),
            "p75_min": g.quantile(0.75),
            "p90_min": g.quantile(0.90),
            "p95_min": g.quantile(0.95),
            "max_min": g.max(),
        }
    ).reset_index()

    return summary.sort_values("mean_min", ascending=False).reset_index(drop=True)


# -----------------------------
# Integrated runner
# -----------------------------
def run_queue_mining(
    event_log_df: pd.DataFrame,
    *,
    state_col: str = "activity",
    covariate_col: Optional[str] = None,
    case_col: str = "case_id",
    time_col: str = "timestamp",
    collapse_consecutive_same_state: bool = False,
    end_rule: Literal["next_event_time", "case_end_time"] = "next_event_time",
    sample_freq: Optional[str] = "15min",
    parse_utc: bool = True,
    debug: bool = False,
    preprocessing_cfg: Optional["CanonicalizationConfig"] = None,
) -> QueueMiningRunResult:
    preprocess_report = None
    df_in = event_log_df.copy()

    # -----------------------------
    # PREPROCESSING SAFETY CHECK + FALLBACK
    # -----------------------------
    if preprocessing_cfg is not None:
        if canonicalize_event_log is None:
            raise ImportError(
                "preprocessing_cfg was provided but event_log_organizer.canonicalize_event_log "
                "could not be imported. Ensure event_log_organizer.py exports canonicalize_event_log."
            )

        # If cfg.case_col doesn't exist, try common fallbacks (visit_id <-> case_id)
        cfg_case = getattr(preprocessing_cfg, "case_col", None)
        if cfg_case and cfg_case not in df_in.columns:
            fallback = None
            if cfg_case == "visit_id" and "case_id" in df_in.columns:
                fallback = "case_id"
            elif cfg_case == "case_id" and "visit_id" in df_in.columns:
                fallback = "visit_id"

            if fallback is not None:
                if debug:
                    print(f"[preprocess] case_col '{cfg_case}' not found. Falling back to '{fallback}'.")
                preprocessing_cfg.case_col = fallback  # type: ignore[attr-defined]
            else:
                raise KeyError(
                    f"Preprocessing case_col='{cfg_case}' not found in dataframe columns. "
                    f"Available columns: {list(df_in.columns)}"
                )

        df_in, preprocess_report = canonicalize_event_log(df_in, cfg=preprocessing_cfg)

    cfg = QueueMiningConfig(
        case_col=case_col,
        time_col=time_col,
        state_col=state_col,
        covariate_col=covariate_col,
        collapse_consecutive_same_state=collapse_consecutive_same_state,
        end_rule=end_rule,
        parse_utc=parse_utc,
        debug=debug,
    )

    intervals_res = build_state_intervals(df_in, cfg=cfg)
    queue_res = compute_queue_lengths(intervals_res, sample_freq=sample_freq)
    summary = summarize_wait_times(intervals_res.intervals, by="state")

    # -----------------------------
    # DROP TERMINAL STATES FROM SUMMARY
    # -----------------------------
    terminal_states = set()
    if preprocessing_cfg is not None:
        terminal_states.update(getattr(preprocessing_cfg, "terminal_steps", ()))
        for step, rule in getattr(preprocessing_cfg, "step_rules", {}).items():
            if getattr(rule, "treat_as_terminal", False):
                terminal_states.add(step)

    if terminal_states:
        summary = summary.loc[~summary["state"].isin(terminal_states)].reset_index(drop=True)

    return QueueMiningRunResult(
        preprocessed_df=df_in,
        preprocess_report=preprocess_report,
        intervals_result=intervals_res,
        queue_result=queue_res,
        wait_summary=summary,
    )


# -----------------------------
# Plotly visualizations
# -----------------------------
def plot_queue_lengths(
    queue_result: QueueLengthResult,
    *,
    use_sampled: bool = True,
    title: str = "Queue length over time (cases in state)",
):
    try:
        import plotly.express as px
    except ImportError as e:
        raise ImportError("plotly is required for visualization. Install with: pip install plotly") from e

    if use_sampled and queue_result.sampled is not None and not queue_result.sampled.empty:
        df = queue_result.sampled.copy()
    else:
        df = queue_result.change_points.copy()
        if df is None or df.empty:
            return px.line(
                pd.DataFrame({"timestamp": [], "queue_length": [], "state": []}),
                x="timestamp",
                y="queue_length",
                color="state",
                title=title,
            )
        df = df[["state", "timestamp", "queue_length"]].copy()

    fig = px.line(df, x="timestamp", y="queue_length", color="state", title=title)
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Cases in state",
        legend_title="State",
        uirevision="queue_lengths",
    )
    return fig


def plot_wait_time_distribution(
    intervals_df: pd.DataFrame,
    *,
    state_col: str = "state",
    duration_col: str = "duration_minutes",
    kind: Literal["box", "violin"] = "box",
    title: str = "Time in state distribution (minutes)",
):
    try:
        import plotly.express as px
    except ImportError as e:
        raise ImportError("plotly is required for visualization. Install with: pip install plotly") from e

    if intervals_df is None or intervals_df.empty:
        return px.box(pd.DataFrame({state_col: [], duration_col: []}), x=state_col, y=duration_col, title=title)

    df = intervals_df.copy()
    df = df[df[duration_col].notna()]

    if kind == "box":
        fig = px.box(df, x=state_col, y=duration_col, points="outliers", title=title)
    elif kind == "violin":
        fig = px.violin(df, x=state_col, y=duration_col, box=True, points="outliers", title=title)
    else:
        raise ValueError("kind must be 'box' or 'violin'.")

    fig.update_layout(xaxis_title="State", yaxis_title="Minutes", uirevision="wait_time_dist")
    return fig


def plot_wait_time_summary(
    summary_df: pd.DataFrame,
    *,
    state_col: str = "state",
    value_col: str = "mean_min",
    title: str = "Average time in state (minutes)",
):
    try:
        import plotly.express as px
    except ImportError as e:
        raise ImportError("plotly is required for visualization. Install with: pip install plotly") from e

    fig = px.bar(summary_df, x=state_col, y=value_col, title=title)
    fig.update_layout(xaxis_title="State", yaxis_title=value_col, uirevision="wait_time_summary")
    return fig
