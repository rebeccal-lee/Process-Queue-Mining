from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd


STANDARD_KEYS = ("case_id", "activity", "timestamp", "resource", "attributes")

@dataclass(frozen=True)
class EventLogMapping:
    """store data with user-defined label mapping"""
    case_id: str
    activity: str
    timestamp: str
    resource: Optional[str] = None
    attributes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["attributes"] = list(self.attributes)
        return d


def _require_columns_exist(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c is not None and c not in df.columns]
    if missing:
        raise KeyError(f"Column(s) not found in dataframe: {missing}. Available: {list(df.columns)}")


def _coerce_timestamp(
    s: pd.Series,
    *,
    utc: bool = False,
    dayfirst: bool = False,
    errors: str = "raise",
) -> pd.Series:
    ts = pd.to_datetime(s, utc=utc, dayfirst=dayfirst, errors=errors)
    if errors == "raise" and ts.isna().any():
        bad = s[ts.isna()].head(10).tolist()
        raise ValueError(f"Failed to parse some timestamps. Examples: {bad}")
    return ts


def build_event_log(
    df: pd.DataFrame,
    mapping: Union[EventLogMapping, Dict[str, Any]],
    *,
    keep_original_names: bool = False,
    standard_names: Dict[str, str] = None,
    drop_missing_core: bool = True,
    timestamp_utc: bool = False,
    timestamp_dayfirst: bool = False,
    timestamp_errors: str = "raise",
    sort_by_time: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    filter and rename dataframe
    """
    if standard_names is None:
        standard_names = {
            "case_id": "case_id",
            "activity": "activity",
            "timestamp": "timestamp",
            "resource": "resource",
        }

    if isinstance(mapping, dict):
        m = EventLogMapping(
            case_id=mapping["case_id"],
            activity=mapping["activity"],
            timestamp=mapping["timestamp"],
            resource=mapping.get("resource"),
            attributes=tuple(mapping.get("attributes", [])),
        )
    else:
        m = mapping

    required_cols = [m.case_id, m.activity, m.timestamp]
    optional_cols = [m.resource] if m.resource else []
    attr_cols = list(m.attributes)

    _require_columns_exist(df, required_cols + optional_cols + attr_cols)

    selected_cols = required_cols + optional_cols + attr_cols
    out = df.loc[:, selected_cols].copy()

    out[m.timestamp] = _coerce_timestamp(
        out[m.timestamp],
        utc=timestamp_utc,
        dayfirst=timestamp_dayfirst,
        errors=timestamp_errors,
    )

    rename_map = {
        m.case_id: standard_names["case_id"],
        m.activity: standard_names["activity"],
        m.timestamp: standard_names["timestamp"],
    }
    if m.resource:
        rename_map[m.resource] = standard_names["resource"]

    out_std = out if keep_original_names else out.rename(columns=rename_map)

    if drop_missing_core:
        core_cols = [
            standard_names["case_id"] if not keep_original_names else m.case_id,
            standard_names["activity"] if not keep_original_names else m.activity,
            standard_names["timestamp"] if not keep_original_names else m.timestamp,
        ]
        out_std = out_std.dropna(subset=core_cols)

    if sort_by_time:
        case_col = standard_names["case_id"] if not keep_original_names else m.case_id
        time_col = standard_names["timestamp"] if not keep_original_names else m.timestamp
        out_std = out_std.sort_values([case_col, time_col], kind="mergesort").reset_index(drop=True)

    # metadata for downstream pipeline
    meta = {
        "user_mapping": m.to_dict(),
        "standard_names": standard_names,
        "rename_map_original_to_standard": rename_map,
        "selected_original_columns": selected_cols,
        "output_columns": list(out_std.columns),
        "n_rows": int(out_std.shape[0]),
    }
    return out_std, meta

def preview_mapping_template(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Visualize dataset mapping
    """
    return {
        "available_columns": list(df.columns),
        "mapping_template": {
            "case_id": None,
            "activity": None,
            "timestamp": None,
            "resource": None,
            "attributes": [],
        },
        "standard_keys": list(STANDARD_KEYS),
    }
