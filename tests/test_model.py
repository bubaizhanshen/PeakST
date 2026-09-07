import numpy as np
import pandas as pd
import torch

from peakst.model import (
    PeakLoss,
    blend_small_cells,
    boundary_continuous_return,
)
from peakst.metrics import evaluate_spectrum
from peakst.reporting import equal_site_equal_seed_summary
from peakst.selection import (
    blocked_reference_folds,
    choose_reference_controls,
    proportional_small_band_rescale,
)
from peakst.training import site_and_class_equal_weights


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


def test_reference_folds_hold_out_complete_days():
    hours = np.arange("2020-01-01", "2020-01-10", dtype="datetime64[h]")
    folds = blocked_reference_folds(hours, n_folds=3)
    assert np.all(np.sum(np.column_stack(folds), axis=1) == 1)
    days = hours.astype("datetime64[D]")
    for fold in folds:
        for day in np.unique(days):
            selected = fold[days == day]
            assert selected.all() or (~selected).all()


def test_proportional_rescale_preserves_within_band_composition():
    cells = np.ones((2, 64))
    cells[0, :11] = np.arange(1.0, 12.0)
    changed = proportional_small_band_rescale(cells, np.log(2.0))
    np.testing.assert_allclose(
        changed[0, :11] / changed[0, :11].sum(),
        cells[0, :11] / cells[0, :11].sum(),
    )
    np.testing.assert_array_equal(changed[:, 11:], cells[:, 11:])


def test_control_selection_prefers_smallest_tied_weight():
    observed = np.ones((20, 64))
    ordinary = observed.copy()
    peak = observed.copy()
    centers = np.geomspace(15.1, 299.0, 64)
    choice = choose_reference_controls(
        observed, ordinary, peak, centers, np.array([0.0, 0.5, 1.0])
    )
    assert choice.fixed_weight == 0.0


def test_gate_weights_balance_sites_and_classes():
    site = np.array(["A", "A", "A", "A", "B", "B"])
    label = np.array([0, 0, 0, 1, 0, 1], dtype=bool)
    weight = site_and_class_equal_weights(site, label)
    np.testing.assert_allclose(
        weight[site == "A"].sum(), weight[site == "B"].sum()
    )
    np.testing.assert_allclose(weight[label].sum(), weight[~label].sum())


def test_site_first_summary_does_not_pool_origins():
    rows = pd.DataFrame(
        {
            "family": ["mlp"] * 3,
            "method": ["m"] * 3,
            "budget_days": [30] * 3,
            "seed": [1] * 3,
            "site": ["A", "A", "B"],
            "origin_id": ["A1", "A2", "B1"],
            "error": [0.0, 0.0, 1.0],
        }
    )
    result = equal_site_equal_seed_summary(rows)
    assert result.loc[0, "error"] == 0.5


def test_reported_raw_and_composition_metrics():
    observed = np.tile(np.arange(1.0, 65.0), (20, 1))
    predicted = observed.copy()
    result = evaluate_spectrum(observed, predicted)
    assert result["small_upper_mae_raw"] == 0.0
    assert result["composition_tv"] == 0.0
    assert result["small_upper_precision"] == 1.0
    assert result["small_upper_recall"] == 1.0
