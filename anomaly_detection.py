# anomaly_detection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config / Results
# ----------------------------

@dataclass(frozen=True)
class AnomalyConfig:
    # Event log schema (standardized)
    case_col: str = "case_id"
    time_col: str = "timestamp"
    act_col: str = "activity"

    # Optional columns
    resource_col: Optional[str] = "resource"  # covariate if you want filters later

    # Time window: define "prediction / monitoring" cutoff (UI slider later)
    cutoff_type: Literal["minutes_since_first_event", "n_events", "full_trace"] = "full_trace"
    cutoff_value: int = 60  # only used for minutes_since_first_event / n_events

    # Long-wait anomaly thresholds
    wait_metric: Literal["case_total_minutes", "activity_dwell_minutes_p95"] = "case_total_minutes"
    z_threshold: float = 3.0              # robust z-score threshold
    quantile_threshold: float = 0.99      # alternative threshold
    use_quantile_threshold: bool = True

    # Sequence anomaly settings
    sequence_method: Literal["bigram_nll", "protocol_deviation"] = "bigram_nll"
    top_k_activities: int = 50
    smoothing: float = 1.0                # Laplace smoothing for bigram model
    seq_quantile_threshold: float = 0.99  # red-flag top 1% most unusual sequences

    # Optional: plug in standard protocol from conformance.py style string
    protocol: Optional[str] = None        # e.g. "Triage -> Registration -> Assessment"
    protocol_mode: Literal["subsequence", "exact", "prefix", "exact_with_extras"] = "subsequence"

    # Cleaning
    sort_events: bool = True
    collapse_consecutive_duplicates: bool = True
    ignore_activities: Optional[Sequence[str]] = None
    max_events_per_case: Optional[int] = None


@dataclass(frozen=True)
class AnomalyResult:
    per_case: pd.DataFrame     # one row per case with anomaly scores + flags
    summary: Dict[str, Any]    # counts / thresholds used
    cfg: AnomalyConfig


# ----------------------------
# Public API
# ----------------------------

def detect_anomalies(
    event_log_df: pd.DataFrame,
    *,
    cfg: Optional[AnomalyConfig] = None,
    # Optional: zone intervals from queue_mining.build_zone_intervals (for dwell-based anomalies)
    zone_intervals_df: Optional[pd.DataFrame] = None,
) -> AnomalyResult:
    """
    Detect "red flag" cases based on:
      1) Extremely long waits / long total case duration (time-based anomalies)
      2) Unusual event sequences (sequence anomalies)

    Returns:
      - per_case dataframe with columns:
          case_id, trace_length, start_time, end_time, case_total_minutes,
          wait_score, wait_flag,
          seq_score, seq_flag,
          red_flag, red_flag_reasons, trace_preview
      - summary dict with thresholds and counts
    """
    if cfg is None:
        cfg = AnomalyConfig()

    required = {cfg.case_col, cfg.time_col, cfg.act_col}
    missing = required - set(event_log_df.columns)
    if missing:
        raise KeyError(f"event_log_df missing required columns: {missing}. Found: {list(event_log_df.columns)}")

    df = event_log_df.copy()
    df[cfg.time_col] = pd.to_datetime(df[cfg.time_col], errors="coerce")
    df = df.dropna(subset=[cfg.time_col])

    if cfg.ignore_activities:
        df = df[~df[cfg.act_col].astype(str).isin(set(cfg.ignore_activities))].copy()

    df[cfg.act_col] = df[cfg.act_col].astype(str)
    if cfg.resource_col and cfg.resource_col in df.columns:
        df[cfg.resource_col] = df[cfg.resource_col].astype(str)

    if cfg.sort_events:
        df = df.sort_values([cfg.case_col, cfg.time_col], kind="mergesort")

    # Apply cutoff slicing (so UI can compute anomalies "as of" a time)
    df_cut = _slice_events_at_cutoff(df, cfg)

    # Optionally cap events per case for performance
    if cfg.max_events_per_case is not None:
        df_cut = df_cut.groupby(cfg.case_col, sort=False).head(cfg.max_events_per_case).copy()

    # Optionally collapse consecutive duplicates (makes sequence scoring more meaningful)
    if cfg.collapse_consecutive_duplicates:
        df_cut = _collapse_consecutive_duplicates(df_cut, cfg)

    # Extract traces for sequence methods and preview
    traces = _extract_traces(df_cut, cfg)

    # Base per-case frame
    per_case = _basic_case_stats(df_cut, traces, cfg)

    # 1) Wait/time anomaly scoring
    wait_scores, wait_flag, wait_thresholds = _compute_wait_anomalies(
        df_cut=df_cut,
        per_case=per_case,
        cfg=cfg,
        zone_intervals_df=zone_intervals_df,
    )
    per_case["wait_score"] = wait_scores
    per_case["wait_flag"] = wait_flag

    # 2) Sequence anomaly scoring
    seq_scores, seq_flag, seq_thresholds = _compute_sequence_anomalies(
        traces=traces,
        cfg=cfg,
        df_for_vocab=df_cut,
    )
    per_case["seq_score"] = seq_scores
    per_case["seq_flag"] = seq_flag

    # Combine into red flag
    per_case["red_flag"] = (per_case["wait_flag"] | per_case["seq_flag"]).astype(bool)
    per_case["red_flag_reasons"] = per_case.apply(_reasons_row, axis=1)

    summary = {
        "n_cases": int(len(per_case)),
        "n_red_flag": int(per_case["red_flag"].sum()),
        "n_wait_flag": int(per_case["wait_flag"].sum()),
        "n_seq_flag": int(per_case["seq_flag"].sum()),
        **wait_thresholds,
        **seq_thresholds,
    }

    # Sort: red flags first, then by combined severity
    per_case["combined_score"] = per_case["wait_score"].fillna(0) + per_case["seq_score"].fillna(0)
    per_case = per_case.sort_values(
        ["red_flag", "combined_score", "case_total_minutes", "trace_length"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    return AnomalyResult(per_case=per_case, summary=summary, cfg=cfg)


# ----------------------------
# Helpers: cutoff + trace extraction
# ----------------------------

def _slice_events_at_cutoff(df: pd.DataFrame, cfg: AnomalyConfig) -> pd.DataFrame:
    if cfg.cutoff_type == "full_trace":
        return df

    if cfg.cutoff_type == "minutes_since_first_event":
        first_time = df.groupby(cfg.case_col, sort=False)[cfg.time_col].transform("min")
        cutoff_ts = first_time + pd.to_timedelta(cfg.cutoff_value, unit="m")
        return df[df[cfg.time_col] <= cutoff_ts].copy()

    if cfg.cutoff_type == "n_events":
        return df.groupby(cfg.case_col, sort=False).head(cfg.cutoff_value).copy()

    raise ValueError(f"Unknown cutoff_type: {cfg.cutoff_type}")


def _collapse_consecutive_duplicates(df: pd.DataFrame, cfg: AnomalyConfig) -> pd.DataFrame:
    # Keep rows where activity changes (or first row in case)
    act = cfg.act_col
    case = cfg.case_col
    prev = df.groupby(case, sort=False)[act].shift(1)
    keep = (prev.isna()) | (df[act] != prev)
    return df[keep].copy()


def _extract_traces(df: pd.DataFrame, cfg: AnomalyConfig) -> Dict[Any, List[str]]:
    case = cfg.case_col
    act = cfg.act_col
    traces: Dict[Any, List[str]] = {}
    for cid, g in df.groupby(case, sort=False):
        traces[cid] = g[act].astype(str).tolist()
    return traces


def _basic_case_stats(df: pd.DataFrame, traces: Dict[Any, List[str]], cfg: AnomalyConfig) -> pd.DataFrame:
    case = cfg.case_col
    t = cfg.time_col

    g = df.groupby(case, sort=False)

    start = g[t].min()
    end = g[t].max()
    n = g.size()

    out = pd.DataFrame({
        case: start.index,
        "start_time": start.values,
        "end_time": end.values,
        "trace_length": n.values.astype(int),
    })
    out["case_total_minutes"] = ((out["end_time"] - out["start_time"]).dt.total_seconds() / 60.0).astype(float)

    # Short preview for UI
    preview = []
    for cid in out[case].tolist():
        tr = traces.get(cid, [])
        preview.append(" → ".join(tr[:12]) + (" …" if len(tr) > 12 else ""))
    out["trace_preview"] = preview

    return out


def _reasons_row(row: pd.Series) -> str:
    reasons = []
    if bool(row.get("wait_flag", False)):
        reasons.append("long_wait")
    if bool(row.get("seq_flag", False)):
        reasons.append("unusual_sequence")
    return ",".join(reasons) if reasons else ""


# ----------------------------
# Wait/time anomalies
# ----------------------------

def _robust_z(x: pd.Series) -> pd.Series:
    # Median / MAD robust z-score
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return 0.6745 * (x - med) / mad


def _compute_wait_anomalies(
    *,
    df_cut: pd.DataFrame,
    per_case: pd.DataFrame,
    cfg: AnomalyConfig,
    zone_intervals_df: Optional[pd.DataFrame],
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Two baseline time-based anomaly options:
      - case_total_minutes: long overall case duration
      - activity_dwell_minutes_p95: flag cases with very long dwell in any activity (requires zone_intervals_df)
    """
    if cfg.wait_metric == "case_total_minutes":
        x = per_case["case_total_minutes"].copy()
        z = _robust_z(x)
        if cfg.use_quantile_threshold:
            q = float(x.quantile(cfg.quantile_threshold))
            flag = x >= q
            thresh_info = {"wait_metric": cfg.wait_metric, "wait_quantile": cfg.quantile_threshold, "wait_threshold_value": q}
        else:
            flag = z >= cfg.z_threshold
            thresh_info = {"wait_metric": cfg.wait_metric, "wait_z_threshold": cfg.z_threshold}

        # Use z as score for ranking (always), even if quantile used for flag
        return z.fillna(0), flag.fillna(False), thresh_info

    if cfg.wait_metric == "activity_dwell_minutes_p95":
        if zone_intervals_df is None:
            raise ValueError("wait_metric='activity_dwell_minutes_p95' requires zone_intervals_df from queue_mining.build_zone_intervals().")

        # Expect columns: case_id, zone/activity, duration_minutes
        case = cfg.case_col
        if case not in zone_intervals_df.columns:
            raise KeyError(f"zone_intervals_df missing {case!r}. Found: {list(zone_intervals_df.columns)}")
        if "duration_minutes" not in zone_intervals_df.columns:
            raise KeyError("zone_intervals_df missing 'duration_minutes'.")

        # For each case, take the max dwell time (most concerning stall)
        stall = (
            zone_intervals_df.groupby(case, sort=False)["duration_minutes"]
            .max()
            .reindex(per_case[case])
        )
        stall = stall.fillna(0).astype(float)

        z = _robust_z(stall)
        if cfg.use_quantile_threshold:
            q = float(stall.quantile(cfg.quantile_threshold))
            flag = stall >= q
            thresh_info = {"wait_metric": cfg.wait_metric, "wait_quantile": cfg.quantile_threshold, "wait_threshold_value": q}
        else:
            flag = z >= cfg.z_threshold
            thresh_info = {"wait_metric": cfg.wait_metric, "wait_z_threshold": cfg.z_threshold}

        return z.fillna(0), flag.fillna(False), thresh_info

    raise ValueError(f"Unknown wait_metric: {cfg.wait_metric}")


# ----------------------------
# Sequence anomalies
# ----------------------------

def _compute_sequence_anomalies(
    *,
    traces: Dict[Any, List[str]],
    cfg: AnomalyConfig,
    df_for_vocab: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Baseline sequence anomaly:
      - bigram_nll: compute negative log-likelihood of each trace under a bigram model
      - protocol_deviation: use simple protocol matching (if cfg.protocol provided)
    """
    case_ids = list(traces.keys())

    if cfg.sequence_method == "protocol_deviation":
        if not cfg.protocol:
            raise ValueError("sequence_method='protocol_deviation' requires cfg.protocol (e.g., 'Triage -> Registration -> Assessment').")

        proto = _parse_protocol(cfg.protocol)
        scores = []
        for cid in case_ids:
            tr = traces[cid]
            ok, missing, extras = _protocol_match(tr, proto, mode=cfg.protocol_mode)
            # score: missing weighs more than extras
            scores.append(float(2 * len(missing) + 0.5 * len(extras)))

        s = pd.Series(scores, index=case_ids, dtype=float)
        thr = float(s.quantile(cfg.seq_quantile_threshold))
        flag = s >= thr
        return s.fillna(0), flag.fillna(False), {
            "seq_method": cfg.sequence_method,
            "seq_quantile": cfg.seq_quantile_threshold,
            "seq_threshold_value": thr,
        }

    if cfg.sequence_method == "bigram_nll":
        # Build vocabulary from df_for_vocab with top_k activities
        top_acts = (
            df_for_vocab[cfg.act_col]
            .astype(str)
            .value_counts()
            .head(cfg.top_k_activities)
            .index
            .tolist()
        )
        vocab = set(top_acts)
        UNK = "__UNK__"
        START = "__START__"
        END = "__END__"

        # Collect bigram counts with smoothing
        # P(next|prev) from counts
        bigram_counts: Dict[Tuple[str, str], float] = {}
        unigram_counts: Dict[str, float] = {}

        def norm_act(a: str) -> str:
            return a if a in vocab else UNK

        for tr in traces.values():
            seq = [START] + [norm_act(a) for a in tr] + [END]
            for i in range(len(seq) - 1):
                prev, nxt = seq[i], seq[i + 1]
                bigram_counts[(prev, nxt)] = bigram_counts.get((prev, nxt), 0.0) + 1.0
                unigram_counts[prev] = unigram_counts.get(prev, 0.0) + 1.0

        # Possible next tokens size for smoothing
        V = len(vocab) + 2  # UNK + END (START is prev only)
        alpha = float(cfg.smoothing)

        def nll_for_trace(tr: List[str]) -> float:
            seq = [START] + [norm_act(a) for a in tr] + [END]
            nll = 0.0
            for i in range(len(seq) - 1):
                prev, nxt = seq[i], seq[i + 1]
                c_big = bigram_counts.get((prev, nxt), 0.0)
                c_prev = unigram_counts.get(prev, 0.0)
                # Laplace smoothing
                p = (c_big + alpha) / (c_prev + alpha * V) if (c_prev + alpha * V) > 0 else 1.0 / V
                nll += -np.log(p)
            # Normalize by length so long traces aren't automatically "anomalous"
            return float(nll / max(1, (len(seq) - 1)))

        scores = [nll_for_trace(traces[cid]) for cid in case_ids]
        s = pd.Series(scores, index=case_ids, dtype=float)

        thr = float(s.quantile(cfg.seq_quantile_threshold))
        flag = s >= thr
        return s.fillna(0), flag.fillna(False), {
            "seq_method": cfg.sequence_method,
            "seq_quantile": cfg.seq_quantile_threshold,
            "seq_threshold_value": thr,
            "seq_top_k_activities": cfg.top_k_activities,
            "seq_smoothing": cfg.smoothing,
        }

    raise ValueError(f"Unknown sequence_method: {cfg.sequence_method}")


def _parse_protocol(protocol: str) -> List[str]:
    # accepts "A -> B -> C" or "A,B,C"
    s = protocol.strip()
    if "->" in s:
        parts = [p.strip() for p in s.split("->")]
    else:
        parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _protocol_match(
    trace: List[str],
    protocol: List[str],
    *,
    mode: Literal["subsequence", "exact", "prefix", "exact_with_extras"] = "subsequence",
) -> Tuple[bool, List[str], List[str]]:
    """
    Returns:
      ok, missing_steps, extra_events

    - subsequence: protocol must appear in order within trace (not necessarily contiguous)
    - prefix: trace must start with protocol (can have extra after)
    - exact: trace must equal protocol (same length + order)
    - exact_with_extras: protocol must appear contiguously; extras allowed before/after
    """
    tr = trace

    if mode == "exact":
        ok = tr == protocol
        missing = [] if ok else [p for p in protocol if p not in tr]
        extras = [] if ok else [a for a in tr if a not in protocol]
        return ok, missing, extras

    if mode == "prefix":
        ok = tr[: len(protocol)] == protocol
        missing = [] if ok else protocol[len([i for i in range(min(len(tr), len(protocol))) if tr[i] == protocol[i]]) :]
        extras = tr[len(protocol):] if len(tr) > len(protocol) else []
        return ok, missing, extras

    if mode == "subsequence":
        j = 0
        matched = []
        for a in tr:
            if j < len(protocol) and a == protocol[j]:
                matched.append(a)
                j += 1
        ok = (j == len(protocol))
        missing = protocol[j:] if not ok else []
        extras = [a for a in tr if a not in protocol]
        return ok, missing, extras

    if mode == "exact_with_extras":
        # protocol must appear contiguously somewhere
        ok = False
        start_idx = -1
        for i in range(0, max(0, len(tr) - len(protocol) + 1)):
            if tr[i:i + len(protocol)] == protocol:
                ok = True
                start_idx = i
                break
        if ok:
            extras = tr[:start_idx] + tr[start_idx + len(protocol):]
            return True, [], extras
        missing = protocol  # if no contiguous match, treat as missing
        extras = [a for a in tr if a not in protocol]
        return False, missing, extras

    raise ValueError(f"Unknown protocol mode: {mode}")
