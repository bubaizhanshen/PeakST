"""Endpoint metrics for full spectra and high small-particle concentrations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def evaluate_spectrum(
    observed: np.ndarray,
    predicted: np.ndarray,
    test_time: np.ndarray | None = None,
    reference_threshold: float | None = None,
    small_cells: int = 11,
    upper_quantile: float = 0.90,
) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("observed and predicted spectra must have equal 2D shapes")
    if not np.isfinite(predicted).all() or np.any(predicted < 0):
        raise ValueError("predictions must be finite and nonnegative")
    observed_small = observed[:, :small_cells].sum(axis=1)
    predicted_small = predicted[:, :small_cells].sum(axis=1)
    threshold = float(np.quantile(observed_small, upper_quantile))
    high = observed_small >= threshold
    predicted_high = predicted_small >= threshold
    false_positive = np.sum(~high & predicted_high)
    true_negative = np.sum(~high & ~predicted_high)
    true_positive = np.sum(high & predicted_high)
    false_negative = np.sum(high & ~predicted_high)
    predicted_positive = np.sum(predicted_high)
    result = {
        "cell_mae_log": float(
            np.abs(np.log1p(predicted) - np.log1p(observed)).mean()
        ),
        "small_all_mae_log": float(
            np.abs(np.log1p(predicted_small) - np.log1p(observed_small)).mean()
        ),
        "small_upper_mae_log": float(
            np.abs(
                np.log1p(predicted_small[high]) - np.log1p(observed_small[high])
            ).mean()
        ),
        "small_upper_mean_ratio": float(
            predicted_small[high].mean() / observed_small[high].mean()
        ),
        "small_upper_precision": float(
            true_positive / predicted_positive if predicted_positive else 0.0
        ),
        "small_upper_recall": float(
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        ),
        "small_upper_false_positive_rate": float(
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else 0.0
        ),
        "small_upper_average_precision": float(
            average_precision_score(high, predicted_small)
        ),
    }
    if reference_threshold is not None:
        reference_high = observed_small >= float(reference_threshold)
        reference_predicted = predicted_small >= float(reference_threshold)
        tp = np.sum(reference_high & reference_predicted)
        fp = np.sum(~reference_high & reference_predicted)
        tn = np.sum(~reference_high & ~reference_predicted)
        fn = np.sum(reference_high & ~reference_predicted)
        result.update(
            {
                "reference_threshold": float(reference_threshold),
                "reference_threshold_observed_hours": int(reference_high.sum()),
                "reference_threshold_predicted_hours": int(reference_predicted.sum()),
                "reference_threshold_precision": float(tp / (tp + fp) if tp + fp else 0.0),
                "reference_threshold_recall": float(tp / (tp + fn) if tp + fn else 0.0),
                "reference_threshold_false_positive_rate": float(
                    fp / (fp + tn) if fp + tn else 0.0
                ),
                "reference_threshold_average_precision": float(
                    average_precision_score(reference_high, predicted_small)
                ),
            }
        )
    if test_time is not None:
        dates = np.asarray(test_time).astype("datetime64[ns]").astype("datetime64[D]")
        daily = pd.DataFrame(
            {"date": dates, "observed": observed_small, "predicted": predicted_small}
        ).groupby("date").max()
        result["small_daily_max_mae_log"] = float(
            np.abs(np.log1p(daily.predicted) - np.log1p(daily.observed)).mean()
        )
    return result
