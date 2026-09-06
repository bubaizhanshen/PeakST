"""PeakST loss and output-combination operations."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PeakLoss(nn.Module):
    """Full-spectrum MSE plus an additional high-state small-cell term."""

    def __init__(self, small_cells: int = 11, additional_weight: float = 4.0):
        super().__init__()
        self.small_cells = int(small_cells)
        self.additional_weight = float(additional_weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        high_state: torch.Tensor,
    ) -> torch.Tensor:
        squared = (prediction - target).square()
        full = squared.mean(dim=(-2, -1))
        small = squared[..., : self.small_cells].mean(dim=(-2, -1))
        return (full + self.additional_weight * high_state * small).mean()


def blend_small_cells(
    general: np.ndarray,
    peak: np.ndarray,
    score: np.ndarray | float,
    small_cells: int = 11,
) -> np.ndarray:
    general = np.asarray(general, dtype=float)
    peak = np.asarray(peak, dtype=float)
    if general.shape != peak.shape:
        raise ValueError("general and peak predictions must have the same shape")
    weight = np.asarray(score, dtype=float)
    if weight.ndim == 0:
        weight = np.full(len(general), float(weight))
    if weight.shape != (len(general),) or np.any((weight < 0) | (weight > 1)):
        raise ValueError("score must contain one value in [0, 1] per hour")
    output = general.copy()
    output[:, :small_cells] += weight[:, None] * (
        peak[:, :small_cells] - general[:, :small_cells]
    )
    return output


def continuation_weights(
    centers_nm: np.ndarray, start_nm: float = 25.0, end_nm: float = 125 / 3
) -> np.ndarray:
    centers = np.asarray(centers_nm, dtype=float)
    weight = np.zeros_like(centers)
    transition = (centers > start_nm) & (centers < end_nm)
    phase = (np.log(centers[transition]) - np.log(start_nm)) / (
        np.log(end_nm) - np.log(start_nm)
    )
    weight[transition] = 0.5 * (1.0 + np.cos(np.pi * phase))
    return weight


def boundary_continuous_return(
    general: np.ndarray,
    blended: np.ndarray,
    centers_nm: np.ndarray,
    small_cells: int = 11,
    start_nm: float = 25.0,
    end_nm: float = 125 / 3,
) -> np.ndarray:
    general = np.asarray(general, dtype=float)
    blended = np.asarray(blended, dtype=float)
    if general.shape != blended.shape:
        raise ValueError("general and blended predictions must have the same shape")
    weight = continuation_weights(centers_nm, start_nm, end_nm)
    output = general.copy()
    output[:, :small_cells] = blended[:, :small_cells]
    residual = np.log1p(blended[:, small_cells - 1]) - np.log1p(
        general[:, small_cells - 1]
    )
    transition = weight > 0
    output[:, transition] = np.expm1(
        np.log1p(general[:, transition])
        + residual[:, None] * weight[None, transition]
    )
    return np.maximum(output, 0.0)
