from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal

import pandas as pd


@dataclass
class StepRule:
    """
    Backend rule template for a particular step/event name.
    Frontend can define any subset of these later.
    """
    drop_if_midnight: bool = False
    drop_if_missing_covariate: bool = False  # e.g. drop if initial_zone missing
    drop_if_timestamp_missing: bool = True   # usually always True (global drop handles this)
    treat_as_terminal: bool = False
    priority_rank: Optional[int] = None


@dataclass
class CanonicalizationConfig:
    case_col: str = "case_id"
    time_col: str = "timestamp"
    event_col: str = "activity"
    covariate_col: Optional[str] = None

    terminal_steps: Tuple[str, ...] = ()
    step_rules: Dict[str, StepRule] = field(default_factory=dict)

    parse_utc: bool = True
    enforce_monotonic_time: bool = True

    max_backward_fraction: float = 0.30
    max_time_repairs_per_case: int = 10

    terminal_fix_mode: Literal[
        "push_seconds",
        "roll_day_if_midnight_and_backwards"
    ] = "push_seconds"

    drop_low_quality_cases: bool = True

    # -----------------------------
    # NEW: "unstable columns" missingness handling
    # -----------------------------
    unstable_columns: Tuple[str, ...] = ()
    unstable_missing_mode: Literal[
        "none",
        "drop_rows",
        "drop_cases",
        "drop_rows_for_steps_only",
    ] = "none"

    # Used only for drop_cases
    unstable_case_missing_threshold: float = 0.30  # fraction of rows in case with missing unstable cols

    # Used only for drop_rows_for_steps_only
    unstable_steps_scope: Tuple[str, ...] = ()

    debug: bool = False


@dataclass
class CanonicalizationReport:
    n_rows_in: int
    n_rows_out: int
    n_cases_in: int
    n_cases_out: int
    dropped_cases: int
    dropped_rows: int

    dropped_midnight_steps: Dict[str, int]
    dropped_missing_covariate_steps: Dict[str, int]

    terminal_reordered_cases: int
    monotonic_repairs_applied: int
    backward_cases_flagged: int

    # -----------------------------
    # NEW: unstable missingness reporting
    # -----------------------------
    unstable_columns_used: Tuple[str, ...] = ()
    unstable_missing_mode: str = "none"
    unstable_case_missing_threshold: float = 0.0
    dropped_unstable_rows: int = 0
    dropped_unstable_cases: int = 0


def _is_midnight(ts: pd.Series) -> pd.Series:
    return (ts.dt.hour == 0) & (ts.dt.minute == 0) & (ts.dt.second == 0)


def canonicalize_event_log(
    df: pd.DataFrame,
    *,
    cfg: CanonicalizationConfig,
) -> tuple[pd.DataFrame, CanonicalizationReport]:
    """
    Canonicalizes an event log:
    - drop missing essentials (case/time/event)
    - apply step-specific drop rules (e.g., drop ambulance arrival at midnight)
    - apply unstable missingness policy (NEW)
    - ensure terminal steps appear last
    - enforce monotonic timestamps within each case (minimal forward shifts)
    - optionally drop very low-quality cases
    """
    out = df.copy()

    required = [cfg.case_col, cfg.time_col, cfg.event_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {list(out.columns)}")

    # Parse timestamps
    out[cfg.time_col] = pd.to_datetime(out[cfg.time_col], errors="coerce", utc=cfg.parse_utc)

    n_rows_in = len(out)
    n_cases_in = out[cfg.case_col].nunique(dropna=True)

    # Drop missing essentials (always)
    out = out.dropna(subset=[cfg.case_col, cfg.time_col, cfg.event_col]).copy()
    out[cfg.event_col] = out[cfg.event_col].astype(str)

    # Build effective terminal set
    terminal_set = set(cfg.terminal_steps)
    for step, rule in cfg.step_rules.items():
        if rule.treat_as_terminal:
            terminal_set.add(step)

    # -----------------------------
    # Step-specific drop rules
    # -----------------------------
    dropped_midnight_steps: Dict[str, int] = {}
    dropped_missing_covariate_steps: Dict[str, int] = {}

    # Drop if midnight for specific steps
    for step, rule in cfg.step_rules.items():
        if rule.drop_if_midnight:
            midnight_mask = _is_midnight(out[cfg.time_col])
            mask = (out[cfg.event_col] == step) & midnight_mask
            count = int(mask.sum())
            if count:
                dropped_midnight_steps[step] = count
                out = out.loc[~mask].copy()

    # Drop if missing covariate for specific steps
    if cfg.covariate_col and cfg.covariate_col in out.columns:
        cov_missing = out[cfg.covariate_col].isna()
        for step, rule in cfg.step_rules.items():
            if rule.drop_if_missing_covariate:
                mask = (out[cfg.event_col] == step) & cov_missing
                count = int(mask.sum())
                if count:
                    dropped_missing_covariate_steps[step] = count
                    out = out.loc[~mask].copy()

    # -----------------------------
    # NEW: Unstable columns missingness policy
    # -----------------------------
    unstable_cols_used = tuple([c for c in cfg.unstable_columns if c in out.columns])
    dropped_unstable_rows = 0
    dropped_unstable_cases = 0

    if cfg.unstable_missing_mode != "none" and unstable_cols_used:
        missing_any = out.loc[:, list(unstable_cols_used)].isna().any(axis=1)

        if cfg.unstable_missing_mode == "drop_rows":
            dropped_unstable_rows = int(missing_any.sum())
            out = out.loc[~missing_any].copy()

        elif cfg.unstable_missing_mode == "drop_rows_for_steps_only":
            if cfg.unstable_steps_scope:
                scope_mask = out[cfg.event_col].isin(cfg.unstable_steps_scope)
                drop_mask = missing_any & scope_mask
                dropped_unstable_rows = int(drop_mask.sum())
                out = out.loc[~drop_mask].copy()
            else:
                # If no scope provided, behave like drop_rows
                dropped_unstable_rows = int(missing_any.sum())
                out = out.loc[~missing_any].copy()

        elif cfg.unstable_missing_mode == "drop_cases":
            # fraction of rows per case that have missing in any unstable col
            frac_missing = (
                out.assign(_miss=missing_any)
                   .groupby(cfg.case_col, sort=False)["_miss"]
                   .mean()
            )
            bad_cases = frac_missing[frac_missing > float(cfg.unstable_case_missing_threshold)].index
            dropped_unstable_cases = int(len(bad_cases))
            if dropped_unstable_cases:
                out = out.loc[~out[cfg.case_col].isin(bad_cases)].copy()

    # De-duplicate identical event records
    out = out.drop_duplicates(subset=[cfg.case_col, cfg.time_col, cfg.event_col], keep="last")

    # -----------------------------
    # Ordering priority (for ties)
    # -----------------------------
    def _rank_for_event(e: str) -> int:
        rule = cfg.step_rules.get(e)
        if rule and rule.priority_rank is not None:
            return int(rule.priority_rank)
        if e in terminal_set:
            return 100
        return 50

    out["_rank"] = out[cfg.event_col].map(_rank_for_event).astype(int)
    out = out.sort_values([cfg.case_col, cfg.time_col, "_rank"], kind="mergesort")

    # -----------------------------
    # Per-case fixes
    # -----------------------------
    terminal_reordered_cases = 0
    monotonic_repairs_applied = 0
    backward_cases_flagged = 0

    def fix_case(g: pd.DataFrame) -> pd.DataFrame:
        nonlocal terminal_reordered_cases, monotonic_repairs_applied, backward_cases_flagged
        g = g.copy()

        diffs = g[cfg.time_col].diff().dt.total_seconds()
        backward = (diffs < 0)
        frac_backward = float(backward.mean()) if len(g) > 1 else 0.0

        if frac_backward > cfg.max_backward_fraction:
            g["_drop_case"] = True
            backward_cases_flagged += 1
            return g

        # Ensure terminal steps end the path
        if terminal_set:
            is_terminal = g[cfg.event_col].isin(terminal_set)
            if is_terminal.any() and (~is_terminal).any():
                last_non_terminal_time = g.loc[~is_terminal, cfg.time_col].max()
                need_fix = is_terminal & (g[cfg.time_col] <= last_non_terminal_time)

                if need_fix.any():
                    terminal_reordered_cases += 1
                    term_rows = g.index[need_fix].tolist()

                    if cfg.terminal_fix_mode == "push_seconds":
                        # stagger terminal pushes to avoid collisions
                        for k, idx in enumerate(term_rows):
                            g.at[idx, cfg.time_col] = last_non_terminal_time + pd.Timedelta(seconds=1 + k)
                    else:
                        mid = _is_midnight(g[cfg.time_col])
                        for k, idx in enumerate(term_rows):
                            if bool(mid.loc[idx]):
                                g.at[idx, cfg.time_col] = g.at[idx, cfg.time_col] + pd.Timedelta(days=1)
                            else:
                                g.at[idx, cfg.time_col] = last_non_terminal_time + pd.Timedelta(seconds=1 + k)

        # Enforce monotonic timestamps (minimal forward shifts)
        if cfg.enforce_monotonic_time and len(g) > 1:
            before = g[cfg.time_col].copy()
            g[cfg.time_col] = g[cfg.time_col].cummax()
            repairs = int((g[cfg.time_col] != before).sum())
            if repairs > cfg.max_time_repairs_per_case:
                g["_drop_case"] = True
                backward_cases_flagged += 1
                return g
            monotonic_repairs_applied += repairs

        g["_drop_case"] = False
        return g

    # Robust groupby apply (keeps case_col safe across pandas versions)
    try:
        out = out.groupby(cfg.case_col, group_keys=True).apply(fix_case, include_groups=False)
    except TypeError:
        out = out.groupby(cfg.case_col, group_keys=True).apply(fix_case)

    if cfg.case_col not in out.columns:
        if isinstance(out.index, pd.MultiIndex):
            out[cfg.case_col] = out.index.get_level_values(0)
        else:
            out[cfg.case_col] = out.index
    out = out.reset_index(drop=True)

    if cfg.drop_low_quality_cases:
        out = out.loc[~out["_drop_case"]].copy()

    out = out.drop(columns=["_rank", "_drop_case"], errors="ignore")

    n_rows_out = len(out)
    n_cases_out = out[cfg.case_col].nunique(dropna=True)

    report = CanonicalizationReport(
        n_rows_in=n_rows_in,
        n_rows_out=n_rows_out,
        n_cases_in=n_cases_in,
        n_cases_out=n_cases_out,
        dropped_cases=n_cases_in - n_cases_out,
        dropped_rows=n_rows_in - n_rows_out,
        dropped_midnight_steps=dropped_midnight_steps,
        dropped_missing_covariate_steps=dropped_missing_covariate_steps,
        terminal_reordered_cases=terminal_reordered_cases,
        monotonic_repairs_applied=monotonic_repairs_applied,
        backward_cases_flagged=backward_cases_flagged,
        unstable_columns_used=unstable_cols_used,
        unstable_missing_mode=str(cfg.unstable_missing_mode),
        unstable_case_missing_threshold=float(cfg.unstable_case_missing_threshold),
        dropped_unstable_rows=int(dropped_unstable_rows),
        dropped_unstable_cases=int(dropped_unstable_cases),
    )

    if cfg.debug:
        print("[canonicalize] report:", report)

    return out.reset_index(drop=True), report
