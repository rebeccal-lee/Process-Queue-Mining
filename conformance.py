from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd

ConformanceMode = Literal[
    "exact", # trace must equal protocol exactly
    "exact_with_extras", # protocol must be a contiguous subsequence of trace
    "subsequence", # protocol must appear in order (gaps allowed)
    "prefix", # trace must start with protocol (exact prefix)
]


@dataclass(frozen=True)
class ConformanceConfig:
    """
    Conformance configuration

    contiguous_requires_start: In "exact_with_extras", allow protocol anywhere as a contiguous block
    (A,B,C consecutive), or require it to start at beginning.

    collapse_consecutive_duplicates: if True, ignore repeated consecutive activities
    (e.g., A,A,A -> A) in both trace + protocol.

    ignore_activities: if set, treat these activity labels as ignorable noise (removed before checking).

    sort_events: if True, perform a stable sort by (case_col, time_col).

    max_events_per_case: if you want to cap trace length for performance in UI (still returns a flag)
    """
    case_col: str = "case_id"
    act_col: str = "activity"
    time_col: str = "timestamp"

    mode: ConformanceMode = "subsequence"
    contiguous_requires_start: bool = False
    collapse_consecutive_duplicates: bool = False
    ignore_activities: Optional[Sequence[str]] = None
    sort_events: bool = True
    max_events_per_case: Optional[int] = None


@dataclass(frozen=True)
class ConformanceResult:
    per_case: pd.DataFrame
    summary: pd.DataFrame
    protocol: List[str]
    config: ConformanceConfig

def parse_protocol(
    protocol: Union[str, Sequence[str]],
    *,
    arrow_tokens: Sequence[str] = ("->", "→"),
) -> List[str]:
    """
    Accepts either:
      - a string like "Triage -> Registration -> Assessment"
      - a list/tuple of strings

    Returns a cleaned list of activity labels,
    """
    if isinstance(protocol, str):
        s = protocol
        for tok in arrow_tokens:
            s = s.replace(tok, "->")
        parts = [p.strip() for p in s.split("->")]
        return [p for p in parts if p]
    else:
        return [str(x).strip() for x in protocol if str(x).strip()]

# Extract trace
def extract_traces(
    event_log_df: pd.DataFrame,
    *,
    case_col: str = "case_id",
    act_col: str = "activity",
    time_col: str = "timestamp",
    sort_events: bool = True,
    ignore_activities: Optional[Sequence[str]] = None,
    collapse_consecutive_duplicates: bool = False,
    max_events_per_case: Optional[int] = None,
) -> Dict[Any, List[str]]:
    """
    Returns dict: case_id -> ordered list of activities.
    """
    required = {case_col, act_col, time_col}
    missing = required - set(event_log_df.columns)
    if missing:
        raise KeyError(f"event_log_df missing required columns: {missing}. Found: {list(event_log_df.columns)}")

    df = event_log_df[[case_col, act_col, time_col]].copy()

    if sort_events:
        df = df.sort_values([case_col, time_col], kind="mergesort")

    if ignore_activities:
        ignore_set = set(map(str, ignore_activities))
        df = df[~df[act_col].astype(str).isin(ignore_set)].copy()

    # Build traces
    traces: Dict[Any, List[str]] = {}
    for cid, grp in df.groupby(case_col, sort=False):
        acts = grp[act_col].astype(str).tolist()

        if collapse_consecutive_duplicates:
            acts = _collapse_consecutive(acts)

        truncated = False
        if max_events_per_case is not None and len(acts) > max_events_per_case:
            acts = acts[:max_events_per_case]
            truncated = True

        traces[cid] = acts

    return traces

def _collapse_consecutive(seq: List[str]) -> List[str]:
    if not seq:
        return seq
    out = [seq[0]]
    for x in seq[1:]:
        if x != out[-1]:
            out.append(x)
    return out



# Conformance
def check_conformance(
    traces: Dict[Any, List[str]],
    protocol: Union[str, Sequence[str]],
    *,
    cfg: Optional[ConformanceConfig] = None,
) -> ConformanceResult:
    """
    traces : dict
        case_id -> list of activity labels
    protocol : str or list(str)
    cfg : ConformanceConfig -> settings (mode, ignore activities, etc.)
    """
    if cfg is None:
        cfg = ConformanceConfig()

    proto = parse_protocol(protocol)
    if cfg.collapse_consecutive_duplicates:
        proto = _collapse_consecutive(proto)

    if not proto:
        raise ValueError("Protocol is empty after parsing/cleaning.")

    rows: List[Dict[str, Any]] = []

    for cid, trace_raw in traces.items():
        trace = list(map(str, trace_raw))
        if cfg.ignore_activities:
            ignore_set = set(map(str, cfg.ignore_activities))
            trace = [a for a in trace if a not in ignore_set]
        if cfg.collapse_consecutive_duplicates:
            trace = _collapse_consecutive(trace)

        # compute conformance + diagnostics
        diag = _diagnose_case(trace, proto, cfg.mode, cfg.contiguous_requires_start)

        rows.append({
            "case_id": cid,
            "conforms": diag["conforms"],
            "mode": cfg.mode,
            "trace_length": len(trace),
            "protocol_length": len(proto),

            "deviation_type": diag["deviation_type"],
            "missing_steps": diag["missing_steps"],
            "missing_count": len(diag["missing_steps"]),
            "extra_events_count": diag["extra_events_count"],
            "matched_positions": diag["matched_positions"],
            "matched_count": len(diag["matched_positions"]),
            "first_deviation_index": diag["first_deviation_index"],
            "trace_preview": trace[:25],
        })

    per_case = pd.DataFrame(rows)

    # summary table
    summary = (
        per_case
        .groupby(["mode", "conforms", "deviation_type"], dropna=False)
        .size()
        .reset_index(name="n_cases")
        .sort_values(["mode", "conforms", "n_cases"], ascending=[True, False, False])
        .reset_index(drop=True)
    )

    return ConformanceResult(per_case=per_case, summary=summary, protocol=proto, config=cfg)


def _diagnose_case(
    trace: List[str],
    proto: List[str],
    mode: ConformanceMode,
    contiguous_requires_start: bool,
) -> Dict[str, Any]:
    """
    Returns dict: conforms, deviation_type, missing_steps, extra_events_count, matched_positions, first_deviation_index
    """
    # Mode: exact
    if mode == "exact":
        if trace == proto:
            return _ok(trace, proto, matched_positions=list(range(len(proto))))
        first = _first_mismatch_index(trace, proto)
        missing, extras = _diff_multiset_like(trace, proto)
        dev = "mismatch"
        if missing:
            dev = "missing_steps"
        elif extras:
            dev = "extra_events"
        return _bad(trace, proto, deviation_type=dev, missing_steps=missing, extra_events_count=extras, matched_positions=[], first_deviation_index=first)

    # Mode: prefix - trace starts with protocol
    if mode == "prefix":
        if len(trace) >= len(proto) and trace[:len(proto)] == proto:
            extras = len(trace) - len(proto)
            dev = "none" if extras == 0 else "extra_events"
            conforms = True
            return {
                "conforms": conforms,
                "deviation_type": "none" if conforms else dev,
                "missing_steps": [],
                "extra_events_count": extras,
                "matched_positions": list(range(len(proto))),
                "first_deviation_index": None,
            }
        matched_positions, missing_steps, order_violation = _align_subsequence(trace, proto)
        first = _first_mismatch_index(trace, proto[: min(len(trace), len(proto))])
        dev = "order_violation" if order_violation else ("missing_steps" if missing_steps else "mismatch")
        extras = max(0, len(trace) - len(proto))
        return _bad(trace, proto, deviation_type=dev, missing_steps=missing_steps, extra_events_count=extras, matched_positions=matched_positions, first_deviation_index=first)

    # Mode: exact_with_extras - protocol is a contiguous block in trace)
    if mode == "exact_with_extras":
        start = _find_contiguous_block(trace, proto, require_start=contiguous_requires_start)
        if start is not None:
            matched = list(range(start, start + len(proto)))
            extras = len(trace) - len(proto)
            return {
                "conforms": True,
                "deviation_type": "none" if extras == 0 else "extra_events",
                "missing_steps": [],
                "extra_events_count": extras,
                "matched_positions": matched,
                "first_deviation_index": None,
            }
        matched_positions, missing_steps, order_violation = _align_subsequence(trace, proto)
        dev = "order_violation" if order_violation else ("missing_steps" if missing_steps else "mismatch")
        extras = max(0, len(trace) - len(proto))
        return _bad(trace, proto, deviation_type=dev, missing_steps=missing_steps, extra_events_count=extras, matched_positions=matched_positions, first_deviation_index=None)

    # Mode: subsequence - protocol in order; gaps allowed
    if mode == "subsequence":
        matched_positions, missing_steps, order_violation = _align_subsequence(trace, proto)
        conforms = (len(missing_steps) == 0)
        extras = max(0, len(trace) - len(matched_positions))
        if conforms:
            dev = "none" if extras == 0 else "extra_events"
        else:
            dev = "order_violation" if order_violation else "missing_steps"
        return {
            "conforms": conforms,
            "deviation_type": dev if not conforms else dev,
            "missing_steps": missing_steps,
            "extra_events_count": extras,
            "matched_positions": matched_positions,
            "first_deviation_index": None,
        }

    raise ValueError(f"Unknown conformance mode: {mode}")

def _ok(trace: List[str], proto: List[str], matched_positions: List[int]) -> Dict[str, Any]:
    extras = max(0, len(trace) - len(matched_positions))
    return {
        "conforms": True,
        "deviation_type": "none" if extras == 0 else "extra_events",
        "missing_steps": [],
        "extra_events_count": extras,
        "matched_positions": matched_positions,
        "first_deviation_index": None,
    }

def _bad(
    trace: List[str],
    proto: List[str],
    *,
    deviation_type: str,
    missing_steps: List[str],
    extra_events_count: int,
    matched_positions: List[int],
    first_deviation_index: Optional[int],
) -> Dict[str, Any]:
    return {
        "conforms": False,
        "deviation_type": deviation_type,
        "missing_steps": missing_steps,
        "extra_events_count": int(extra_events_count),
        "matched_positions": matched_positions,
        "first_deviation_index": first_deviation_index,
    }

# Helper functions
def _align_subsequence(trace: List[str], proto: List[str]) -> Tuple[List[int], List[str], bool]:
    """
    Returns:
      matched_positions: positions in trace where each proto step matched (len <= len(proto))
      missing_steps: steps not matched
      order_violation: True if a missing proto step actually appears in trace but couldn't be matched due to ordering
    """
    matched_positions: List[int] = []
    missing_steps: List[str] = []

    j = 0  # pointer in trace
    last_match = -1

    for step in proto:
        found_at = None
        for k in range(j, len(trace)):
            if trace[k] == step:
                found_at = k
                break
        if found_at is None:
            missing_steps.append(step)
        else:
            matched_positions.append(found_at)
            j = found_at + 1
            last_match = found_at

    # Order violation:
    # If a step is missing but *does* exist somewhere in the trace,
    # interpret as potential out-of-order or repetition issues
    trace_set = set(trace)
    order_violation = any((m in trace_set) for m in missing_steps)

    return matched_positions, missing_steps, order_violation


def _find_contiguous_block(trace: List[str], proto: List[str], *, require_start: bool) -> Optional[int]:
    """
    Finds start index where proto appears as a contiguous block in trace.
    Returns start index or None.
    """
    if not proto:
        return None
    if require_start:
        return 0 if len(trace) >= len(proto) and trace[:len(proto)] == proto else None

    L = len(proto)
    if L > len(trace):
        return None
    for i in range(0, len(trace) - L + 1):
        if trace[i:i+L] == proto:
            return i
    return None


def _first_mismatch_index(a: List[str], b: List[str]) -> Optional[int]:
    """
    Returns first index where sequences differ (0-based), or None if equal up to min length.
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def _diff_multiset_like(trace: List[str], proto: List[str]) -> Tuple[List[str], int]:
    """
      missing_steps = elements in proto not 'covered' by trace counts (multiset-like)
      extra_events_count = number of trace events beyond proto counts
    """
    from collections import Counter
    c_trace = Counter(trace)
    c_proto = Counter(proto)

    missing: List[str] = []
    for k, v in c_proto.items():
        if c_trace[k] < v:
            missing.extend([k] * (v - c_trace[k]))

    # extra count: trace counts that exceed proto counts
    extras = 0
    for k, v in c_trace.items():
        extras += max(0, v - c_proto.get(k, 0))

    return missing, extras
