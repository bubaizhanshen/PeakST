"""PNSD representation and bin-width-aware mapping operations."""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def log10_midpoint_edges(diameter_nm: np.ndarray) -> np.ndarray:
    diameter = np.asarray(diameter_nm, dtype=float)
    if (
        diameter.ndim != 1
        or len(diameter) < 2
        or not np.isfinite(diameter).all()
        or np.any(diameter <= 0)
        or np.any(np.diff(diameter) <= 0)
    ):
        raise ValueError("diameter midpoints must be positive and increasing")
    center = np.log10(diameter)
    edges = np.empty(len(center) + 1)
    edges[1:-1] = 0.5 * (center[:-1] + center[1:])
    edges[0] = center[0] - 0.5 * (center[1] - center[0])
    edges[-1] = center[-1] + 0.5 * (center[-1] - center[-2])
    return edges


def edge_aligned_grid(
    boundaries_nm: Iterable[float], cells_per_band: Iterable[int]
) -> np.ndarray:
    boundaries = np.asarray(tuple(boundaries_nm), dtype=float)
    counts = tuple(int(value) for value in cells_per_band)
    if len(counts) != len(boundaries) - 1 or min(counts) < 1:
        raise ValueError("one positive cell count is required per interval")
    pieces = []
    for index, (lower, upper, count) in enumerate(
        zip(boundaries[:-1], boundaries[1:], counts)
    ):
        part = np.geomspace(lower, upper, count + 1)
        pieces.append(part if index == 0 else part[1:])
    edges = np.concatenate(pieces)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("common edges must be strictly increasing")
    return edges


def density_log10(
    spectrum: np.ndarray,
    semantics: str,
    native_delta_log10_dp: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(spectrum, dtype=float)
    if semantics == "dndlog10dp":
        density = values.copy()
    elif semantics == "dndln_dp":
        density = values * math.log(10.0)
    elif semantics == "bin_concentration":
        if native_delta_log10_dp is None:
            raise ValueError("bin concentration requires native log10 widths")
        width = np.asarray(native_delta_log10_dp, dtype=float)
        if values.shape[-1] != len(width) or np.any(width <= 0):
            raise ValueError("invalid native width vector")
        density = values / width
    else:
        raise ValueError(f"unsupported PNSD semantics: {semantics}")
    density[~np.isfinite(density) | (density < 0)] = np.nan
    return density


def integrate_piecewise_linear_density(
    density: np.ndarray,
    diameter_nm: np.ndarray,
    target_edges_nm: np.ndarray,
) -> np.ndarray:
    values = np.atleast_2d(np.asarray(density, dtype=float))
    source_x = np.log10(np.asarray(diameter_nm, dtype=float))
    target_edges = np.log10(np.asarray(target_edges_nm, dtype=float))
    target_centers = 0.5 * (target_edges[:-1] + target_edges[1:])
    width = np.diff(target_edges)
    mapped = np.full((len(values), len(target_centers)), np.nan)
    for row_index, row in enumerate(values):
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        x = source_x[finite]
        y = row[finite]
        inside = (target_centers >= x[0]) & (target_centers <= x[-1])
        mapped[row_index, inside] = np.interp(target_centers[inside], x, y)
    return mapped * width


def partition_integrated_cells(
    source_counts: np.ndarray,
    source_edges_nm: np.ndarray,
    target_edges_nm: np.ndarray,
) -> np.ndarray:
    """Partition already integrated cells by logarithmic-diameter overlap."""

    counts = np.atleast_2d(np.asarray(source_counts, dtype=float))
    source = np.log10(np.asarray(source_edges_nm, dtype=float))
    target = np.log10(np.asarray(target_edges_nm, dtype=float))
    if counts.shape[1] != len(source) - 1:
        raise ValueError("source count shape does not match source edges")
    overlap = np.maximum(
        0.0,
        np.minimum(source[1:, None], target[None, 1:])
        - np.maximum(source[:-1, None], target[None, :-1]),
    )
    weights = overlap / np.diff(source)[:, None]
    covered = overlap.sum(axis=1) > 0
    if not np.allclose(weights[covered].sum(axis=1), 1.0, atol=1e-12, rtol=0):
        raise ValueError("target grid does not cover every included source cell")
    return counts @ weights
