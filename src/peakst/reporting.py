"""Aggregation helpers matching the manuscript's site-first summaries."""
from __future__ import annotations

import pandas as pd


IDENTIFIERS = {"family", "method", "budget_days", "seed", "site", "origin_id"}


def equal_site_equal_seed_summary(origin_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average origins within site, sites within seed, and then seeds equally."""

    required = IDENTIFIERS
    missing = required.difference(origin_metrics.columns)
    if missing:
        raise ValueError(f"origin metric table is missing: {sorted(missing)}")
    metrics = [
        column
        for column in origin_metrics.columns
        if column not in IDENTIFIERS
        and pd.api.types.is_numeric_dtype(origin_metrics[column])
    ]
    keys = ["family", "method", "budget_days"]
    site_seed = origin_metrics.groupby(
        [*keys, "seed", "site"], as_index=False
    )[metrics].mean()
    seed = site_seed.groupby([*keys, "seed"], as_index=False)[metrics].mean()
    result = seed.groupby(keys, as_index=False)[metrics].mean()
    counts = seed.groupby(keys, as_index=False).seed.nunique().rename(
        columns={"seed": "n_seeds"}
    )
    return result.merge(counts, on=keys, how="left", validate="one_to_one")
