from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Tuple

import pandas as pd

@dataclass(frozen=True)
class QueueMiningConfig:
    case_col: str = "case_id"
    time_col: str = "timestamp"
    zone_col: str = "initial_zone"   # resource/zone column to group queues by

    act_col: Optional[str] = "activity"
    sort_events: bool = True
    drop_missing_zone: bool = True
    collapse_consecutive_same_zone: bool = True

    # Interval end rule: how to define leaving a zone
    end_rule: Literal["next_event_time", "case_end_time"] = "next_event_time"

    max_events_per_case: Optional[int] = None


@dataclass(frozen=True)
class ZoneIntervalsResult:
    intervals: pd.DataFrame  # per-case-per-zone intervals
    config: QueueMiningConfig

@dataclass(frozen=True)
class QueueLengthResult:
    change_points: pd.DataFrame  # step function changes
    sampled: Optional[pd.DataFrame]  # optional regular time grid
    config: QueueMiningConfig

def build_zone_intervals(
    event_log_df: pd.DataFrame,
    *,
    cfg: Optional[QueueMiningConfig] = None,
) -> ZoneIntervalsResult:
    """
    Build per-case intervals of "being in a zone/area of hospital".
    - We assume zone/resource at an event indicates the case is in that zone at that time.
    - Interval starts at event time and ends at next event time (or case end time).

    Output:
      case_id, zone, start_time, end_time, duration_seconds, duration_minutes
      + optional start_activity, end_activity (if act_col exists)
    """
    if cfg is None:
        cfg = QueueMiningConfig()

    required = {cfg.case_col, cfg.time_col, cfg.zone_col}
    missing = required - set(event_log_df.columns)
    if missing:
        raise KeyError(f"event_log_df missing required columns: {missing}. Found: {list(event_log_df.columns)}")

    df = event_log_df.copy()

    # Ensure datetime
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col])

    # Crop missing zone
    if cfg.drop_missing_zone:
        df = df.dropna(subset=[cfg.zone_col])
    df[cfg.zone_col] = df[cfg.zone_col].astype(str)

    # Sort events
    if cfg.sort_events:
        df = df.sort_values([cfg.case_col, cfg.time_col], kind="mergesort")

    if cfg.max_events_per_case is not None:
        df = (
            df.groupby(cfg.case_col, sort=False)
              .head(cfg.max_events_per_case)
              .copy()
        )

    # Next event time and next zone
    df["_next_time"] = df.groupby(cfg.case_col, sort=False)[cfg.time_col].shift(-1)
    df["_next_zone"] = df.groupby(cfg.case_col, sort=False)[cfg.zone_col].shift(-1)

    if cfg.act_col and cfg.act_col in df.columns:
        df["_act"] = df[cfg.act_col].astype(str)
        df["_next_act"] = df.groupby(cfg.case_col, sort=False)["_act"].shift(-1)
    else:
        df["_act"] = None
        df["_next_act"] = None

    # Only keep rows where zone changes (or last row)
    if cfg.collapse_consecutive_same_zone:
        keep = (df[cfg.zone_col] != df["_next_zone"]) | df["_next_zone"].isna()
        df = df[keep].copy()

    # Define end time
    if cfg.end_rule == "next_event_time":
        df["_end_time"] = df["_next_time"]
    elif cfg.end_rule == "case_end_time":
        last_time = df.groupby(cfg.case_col, sort=False)[cfg.time_col].transform("max")
        df["_end_time"] = last_time
    else:
        raise ValueError(f"Unknown end_rule: {cfg.end_rule}")

    # Drop intervals with missing end_time
    df = df.dropna(subset=["_end_time"])

    intervals = pd.DataFrame({
        cfg.case_col: df[cfg.case_col].values,
        "zone": df[cfg.zone_col].values,
        "start_time": df[cfg.time_col].values,
        "end_time": df["_end_time"].values,
        "start_activity": df["_act"].values,
        "end_activity": df["_next_act"].values,
    })

    # Duration
    dt = (intervals["end_time"] - intervals["start_time"])
    intervals["duration_seconds"] = dt.dt.total_seconds().astype(float)
    intervals = intervals[intervals["duration_seconds"] >= 0].copy()
    intervals["duration_minutes"] = intervals["duration_seconds"] / 60.0
    intervals["date"] = intervals["start_time"].dt.date.astype(str)

    return ZoneIntervalsResult(intervals=intervals.reset_index(drop=True), config=cfg)

# Queue length
def compute_queue_length_change_points(
    intervals_df: pd.DataFrame,
    *,
    zone_col: str = "zone",
    start_col: str = "start_time",
    end_col: str = "end_time",
) -> pd.DataFrame:
    """
    Computes step-function queue length change points per zone:
      +1 at start_time, -1 at end_time; cumulative sum yields "cases in zone".

    Return: zone, timestamp, delta, queue_length
    """
    # Start events
    starts = intervals_df[[zone_col, start_col]].copy()
    starts = starts.rename(columns={start_col: "timestamp"})
    starts["delta"] = 1

    # End events
    ends = intervals_df[[zone_col, end_col]].copy()
    ends = ends.rename(columns={end_col: "timestamp"})
    ends["delta"] = -1

    changes = pd.concat([starts, ends], ignore_index=True)
    changes["timestamp"] = pd.to_datetime(changes["timestamp"], errors="coerce")
    changes = changes.dropna(subset=["timestamp"])

    # Sort and cumulative sum per zone
    changes = changes.sort_values([zone_col, "timestamp", "delta"], kind="mergesort")
    changes["queue_length"] = changes.groupby(zone_col, sort=False)["delta"].cumsum()

    # Rename zone col
    changes = changes.rename(columns={zone_col: "zone"})
    return changes.reset_index(drop=True)


def sample_queue_lengths(
    change_points_df: pd.DataFrame,
    *,
    freq: str = "15min",
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Converts change points to time grid per zone

    Returns: zone, timestamp, queue_length
    """
    cp = change_points_df.copy()
    cp["timestamp"] = pd.to_datetime(cp["timestamp"])
    if start is None:
        start = cp["timestamp"].min()
    if end is None:
        end = cp["timestamp"].max()

    grid = pd.date_range(start=start, end=end, freq=freq)
    out_rows: List[pd.DataFrame] = []

    for zone, g in cp.groupby("zone", sort=False):
        g = g.sort_values("timestamp")
        g = g.groupby("timestamp", as_index=False)["queue_length"].last()

        df_grid = pd.DataFrame({"timestamp": grid})
        df_grid = df_grid.merge(g, on="timestamp", how="left")
        df_grid["queue_length"] = df_grid["queue_length"].ffill().fillna(0).astype(float)
        df_grid["zone"] = zone
        out_rows.append(df_grid)

    sampled = pd.concat(out_rows, ignore_index=True)
    return sampled[["zone", "timestamp", "queue_length"]]


def compute_queue_lengths(
    intervals_result: ZoneIntervalsResult,
    *,
    sample_freq: Optional[str] = "15min",
) -> QueueLengthResult:
    """
    helper: intervals -> change points -> (optional) sampled grid.
    """
    cp = compute_queue_length_change_points(intervals_result.intervals)
    sampled = None
    if sample_freq is not None:
        sampled = sample_queue_lengths(cp, freq=sample_freq)
    return QueueLengthResult(change_points=cp, sampled=sampled, config=intervals_result.config)


# Waiting time
def summarize_wait_times(
    intervals_df: pd.DataFrame,
    *,
    by: str = "zone",
    duration_col: str = "duration_minutes",
) -> pd.DataFrame:
    """
    Summary stats of time-in-zone
    """
    g = intervals_df.groupby(by, dropna=False)[duration_col]
    summary = pd.DataFrame({
        "n_intervals": g.size(),
        "mean_min": g.mean(),
        "median_min": g.median(),
        "p75_min": g.quantile(0.75),
        "p90_min": g.quantile(0.90),
        "p95_min": g.quantile(0.95),
        "max_min": g.max(),
    }).reset_index()

    summary = summary.sort_values("mean_min", ascending=False).reset_index(drop=True)
    return summary



# Plotly visualizations

def plot_queue_lengths(
    queue_result: QueueLengthResult,
    *,
    use_sampled: bool = True,
    title: str = "Queue length over time (cases in zone)",
):
    """
    Plotly line chart of queue length over time by zone.
    """
    try:
        import plotly.express as px
    except ImportError as e:
        raise ImportError("plotly is required for visualization. Install with: pip install plotly") from e

    if use_sampled:
        if queue_result.sampled is None:
            raise ValueError("No sampled data available. Recompute with sample_freq != None.")
        df = queue_result.sampled.copy()
    else:
        df = queue_result.change_points.rename(columns={"timestamp": "timestamp"}).copy()
        df = df[["zone", "timestamp", "queue_length"]]

    fig = px.line(
        df,
        x="timestamp",
        y="queue_length",
        color="zone",
        title=title,
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Queue length",
        legend_title="Zone",
        uirevision="queue_lengths",
    )
    return fig


def plot_wait_time_distribution(
    intervals_df: pd.DataFrame,
    *,
    zone_col: str = "zone",
    duration_col: str = "duration_minutes",
    kind: Literal["box", "violin"] = "box",
    title: str = "Time in zone distribution (minutes)",
):
    """
    Plotly distribution of time in zone by zone (box or violin).
    """
    try:
        import plotly.express as px
    except ImportError as e:
        raise ImportError("plotly is required for visualization. Install with: pip install plotly") from e

    df = intervals_df.copy()
    df = df[df[duration_col].notna()]

    if kind == "box":
        fig = px.box(df, x=zone_col, y=duration_col, points="outliers", title=title)
    elif kind == "violin":
        fig = px.violin(df, x=zone_col, y=duration_col, box=True, points="outliers", title=title)
    else:
        raise ValueError("kind must be 'box' or 'violin'.")

    fig.update_layout(
        xaxis_title="Zone",
        yaxis_title="Minutes",
        uirevision="wait_time_dist",
    )
    return fig


def plot_wait_time_summary(
    summary_df: pd.DataFrame,
    *,
    zone_col: str = "zone",
    value_col: str = "mean_min",
    title: str = "Average time in zone (minutes)",
):
    """
    Plotly bar chart of wait time summary (mean by default).
    """
    try:
        import plotly.express as px
    except ImportError as e:
        raise ImportError("plotly is required for visualization. Install with: pip install plotly") from e

    fig = px.bar(summary_df, x=zone_col, y=value_col, title=title)
    fig.update_layout(
        xaxis_title="Zone",
        yaxis_title=value_col,
        uirevision="wait_time_summary",
    )
    return fig
