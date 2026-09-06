import numpy as np

from peakst.remap import (
    edge_aligned_grid,
    integrate_piecewise_linear_density,
    partition_integrated_cells,
)


def test_constant_density_is_resolution_invariant():
    target = edge_aligned_grid((15.0, 25.0, 100.0, 300.0), (11, 30, 23))
    expected = 1000.0 * np.log10(300.0 / 15.0)
    for channels in (23, 39, 54, 103):
        diameter = np.geomspace(10.0, 500.0, channels)
        density = np.full((1, channels), 1000.0)
        mapped = integrate_piecewise_linear_density(density, diameter, target)
        assert np.isclose(mapped.sum(), expected, rtol=0, atol=1e-10)


def test_overlap_partition_conserves_integrated_number():
    source_edges = np.geomspace(15.0, 300.0, 65)
    target_edges = source_edges.copy()
    counts = np.arange(1.0, 65.0)[None, :]
    mapped = partition_integrated_cells(counts, source_edges, target_edges)
    np.testing.assert_allclose(mapped, counts, rtol=0, atol=1e-12)
