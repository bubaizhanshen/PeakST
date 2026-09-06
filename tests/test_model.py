import numpy as np
import torch

from peakst.model import (
    PeakLoss,
    blend_small_cells,
    boundary_continuous_return,
)


def test_peak_loss_adds_small_cell_penalty_only_for_high_states():
    prediction = torch.ones((2, 1, 64))
    target = torch.zeros((2, 1, 64))
    loss = PeakLoss()(prediction, target, torch.tensor([0.0, 1.0]))
    assert torch.isclose(loss, torch.tensor(3.0))


def test_blend_preserves_cells_above_small_range():
    general = np.ones((3, 64))
    peak = np.full((3, 64), 2.0)
    blended = blend_small_cells(general, peak, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(blended[:, 11:], general[:, 11:])


def test_boundary_return_is_nonnegative_and_preserves_small_cells():
    edges = np.concatenate(
        [
            np.geomspace(15.0, 25.0, 12),
            np.geomspace(25.0, 100.0, 31)[1:],
            np.geomspace(100.0, 300.0, 24)[1:],
        ]
    )
    centers = np.sqrt(edges[:-1] * edges[1:])
    general = np.ones((2, 64))
    blended = general.copy()
    blended[:, :11] = 4.0
    output = boundary_continuous_return(general, blended, centers)
    np.testing.assert_array_equal(output[:, :11], blended[:, :11])
    assert np.all(output >= 0)
