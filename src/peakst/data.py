"""Input validation for locally supplied PeakST experiment archives."""
from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "source_x",
    "source_y",
    "source_site",
    "source_year",
    "reference_x",
    "reference_y",
    "reference_time",
    "test_x",
    "test_y",
    "test_time",
    "edges_nm",
)


def load_experiment(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        missing = sorted(set(REQUIRED_KEYS) - set(archive.files))
        if missing:
            raise ValueError(f"experiment archive is missing keys: {missing}")
        data = {key: archive[key].copy() for key in REQUIRED_KEYS}
    if data["source_y"].shape[1] != 64:
        raise ValueError("source_y must contain 64 integrated common cells")
    if data["reference_y"].shape[1] != 64 or data["test_y"].shape[1] != 64:
        raise ValueError("reference_y and test_y must contain 64 common cells")
    if data["source_x"].shape[1] != data["reference_x"].shape[1]:
        raise ValueError("source and reference predictor dimensions differ")
    if data["source_x"].shape[1] != data["test_x"].shape[1]:
        raise ValueError("source and test predictor dimensions differ")
    reference_time = data["reference_time"].astype("datetime64[ns]")
    test_time = data["test_time"].astype("datetime64[ns]")
    if reference_time.max() >= test_time.min():
        raise ValueError("target-reference timestamps must precede the test period")
    if np.any(data["source_y"] < 0) or np.any(data["reference_y"] < 0):
        raise ValueError("PNSD cell concentrations must be nonnegative")
    return data
