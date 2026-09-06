"""Selection rules that operate only on source or target-reference data."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import blend_small_cells, boundary_continuous_return


def blocked_reference_folds(times: np.ndarray, n_folds: int = 3) -> list[np.ndarray]:
    """Return chronological day-block validation masks for a reference set."""

    values = np.asarray(times).astype("datetime64[ns]")
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("reference times must be a nonempty one-dimensional array")
    days = values.astype("datetime64[D]")
    unique_days = np.unique(days)
    if len(unique_days) < n_folds:
        raise ValueError("reference set has fewer days than blocked folds")
    masks = [np.isin(days, block) for block in np.array_split(unique_days, n_folds)]
    if any(not mask.any() or mask.all() for mask in masks):
        raise RuntimeError("invalid blocked reference fold")
    return masks


def proportional_small_band_rescale(
    cells: np.ndarray, log_offset: float, small_cells: int = 11
) -> np.ndarray:
    """Correct the integrated small band and return it proportionally to cells."""

    cells = np.asarray(cells, dtype=float)
    result = cells.copy()
    old_total = cells[:, :small_cells].sum(axis=1)
    new_total = np.expm1(
        np.clip(np.log1p(old_total) + float(log_offset), 0.0, 20.0)
    )
    multiplier = np.divide(
        new_total,
        old_total,
        out=np.ones_like(new_total),
        where=old_total > 0,
    )
    result[:, :small_cells] *= multiplier[:, None]
    return result


@dataclass(frozen=True)
class ReferenceControlChoice:
    log_offset: float
    fixed_weight: float
    ordinary_cell_mae_log: float
    selected_cell_mae_log: float
    selected_upper_mae_log: float


def choose_reference_controls(
    observed: np.ndarray,
    ordinary_oof: np.ndarray,
    peak_oof: np.ndarray,
    centers_nm: np.ndarray,
    weight_grid: np.ndarray,
    maximum_full_spectrum_cost: float = 0.01,
    upper_quantile: float = 0.90,
) -> ReferenceControlChoice:
    """Choose bias and fixed allocation from blocked OOF reference predictions.

    The caller is responsible for producing predictions for which each
    reference row was held out from fitting.  The function never accepts test
    data, and ties in the feasible fixed-weight grid favor the smaller weight.
    """

    observed = np.asarray(observed, dtype=float)
    ordinary_oof = np.asarray(ordinary_oof, dtype=float)
    peak_oof = np.asarray(peak_oof, dtype=float)
    if observed.shape != ordinary_oof.shape or observed.shape != peak_oof.shape:
        raise ValueError("observed and OOF predictions must have equal shapes")
    if observed.shape[1] != 64:
        raise ValueError("the paper workflow requires 64 target cells")
    observed_small = observed[:, :11].sum(axis=1)
    ordinary_small = ordinary_oof[:, :11].sum(axis=1)
    log_offset = float(
        np.median(np.log1p(observed_small) - np.log1p(ordinary_small))
    )
    high = observed_small >= np.quantile(observed_small, upper_quantile)
    ordinary_error = float(
        np.abs(np.log1p(ordinary_oof) - np.log1p(observed)).mean()
    )
    rows = []
    for weight in np.asarray(weight_grid, dtype=float):
        hard = blend_small_cells(ordinary_oof, peak_oof, float(weight))
        prediction = boundary_continuous_return(ordinary_oof, hard, centers_nm)
        cell_error = float(
            np.abs(np.log1p(prediction) - np.log1p(observed)).mean()
        )
        predicted_small = prediction[:, :11].sum(axis=1)
        upper_error = float(
            np.abs(
                np.log1p(predicted_small[high]) - np.log1p(observed_small[high])
            ).mean()
        )
        rows.append((float(weight), cell_error, upper_error))
    feasible = [
        row
        for row in rows
        if row[1] <= ordinary_error * (1.0 + maximum_full_spectrum_cost)
    ]
    if not feasible:
        feasible = [min(rows, key=lambda row: (row[1], row[0]))]
    selected = min(feasible, key=lambda row: (row[2], row[0]))
    return ReferenceControlChoice(
        log_offset=log_offset,
        fixed_weight=selected[0],
        ordinary_cell_mae_log=ordinary_error,
        selected_cell_mae_log=selected[1],
        selected_upper_mae_log=selected[2],
    )
