"""Endpoint metrics for full spectra and high small-particle concentrations."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def evaluate_spectrum(
    observed: np.ndarray,
    predicted: np.ndarray,
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
    predicted_positive = np.sum(predicted_high)
    return {
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
        "small_upper_false_positive_rate": float(
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else 0.0
        ),
        "small_upper_average_precision": float(
            average_precision_score(high, predicted_small)
        ),
    }
