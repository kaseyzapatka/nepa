"""Writer-agnostic, idempotent finalization of the `duration_days` column.

Single source of truth for how `duration_days` is derived on the canonical
`timeline_project_dates.parquet`. Any step that mutates the final date columns
(05_select_dates.py after selection/imputation/order-normalization, and
06_adjudicate_llm.py after injecting adjudication-recovered dates) calls this so
the numeric duration never drifts out of sync with the dates it summarizes.

Background: 06's apply step recomputed `timeline_status` after injecting a
recovered day-level date but never recomputed `duration_days`, leaving ~2,849
complete day/day rows with a null duration (the "stale duration_days" bug fixed
2026-07-24). This pass re-derives the column for ALL rows from the final dates,
so it repairs those rows and stays correct no matter which writer ran last.

Gate (mirrors 05_select_dates.py exactly, lines ~1315-1326):
  duration_days = (decision_date - initiation_date).days
    ONLY when: both dates present
             AND initiation_date_granularity == "day"
             AND decision_date_granularity   == "day"
             AND timeline_status != "invalid_order"
             AND decision_date >= initiation_date
  else: null.

Month/year-granularity endpoints (exact day counts undefined) and invalid-order
rows are therefore left null by design.
"""

from __future__ import annotations

import pandas as pd


def finalize_duration_days(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute `duration_days` for every row in-place under the 05 gate.

    Vectorized and idempotent — running it twice yields the same result because
    the value is always re-derived from the final date columns. Mutates and also
    returns `df` for convenience.
    """
    if df is None or len(df) == 0:
        if df is not None and "duration_days" not in df.columns:
            df["duration_days"] = pd.Series(dtype="float64")
        return df

    required = {
        "initiation_date",
        "decision_date",
        "initiation_date_granularity",
        "decision_date_granularity",
        "timeline_status",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"finalize_duration_days: missing required column(s): {sorted(missing)}"
        )

    init = pd.to_datetime(df["initiation_date"], errors="coerce")
    dec = pd.to_datetime(df["decision_date"], errors="coerce")

    gate = (
        init.notna()
        & dec.notna()
        & (df["initiation_date_granularity"] == "day")
        & (df["decision_date_granularity"] == "day")
        & (df["timeline_status"] != "invalid_order")
        & (dec >= init)
    )

    duration = (dec - init).dt.days  # nullable Int; NaT-derived rows are NA
    # `.where(gate)` nulls every row failing the gate; result is float64 to match
    # the existing DOUBLE column dtype (and preserve NaN for the null rows).
    df["duration_days"] = duration.where(gate).astype("float64")
    return df
