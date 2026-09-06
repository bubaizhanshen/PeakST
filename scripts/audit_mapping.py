#!/usr/bin/env python3
"""Audit a locally supplied native-spectrum representation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from peakst.remap import (
    density_log10,
    edge_aligned_grid,
    integrate_piecewise_linear_density,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    diameter_config = config["diameter"]
    boundaries = (
        diameter_config["lower_nm"],
        diameter_config["small_upper_nm"],
        diameter_config["ultrafine_upper_nm"],
        diameter_config["upper_nm"],
    )
    edges = edge_aligned_grid(boundaries, diameter_config["cells_per_band"])
    with np.load(args.input, allow_pickle=False) as archive:
        diameter = archive["diameter_midpoint_nm"].astype(float)
        spectrum = archive["spectrum"].astype(float)
        valid = archive["valid_hour"].astype(bool)
        semantics = str(archive["semantics"].item())
        widths = (
            archive["native_delta_log10_dp"].astype(float)
            if "native_delta_log10_dp" in archive.files
            else None
        )
    density = density_log10(spectrum, semantics, widths)
    mapped = integrate_piecewise_linear_density(density, diameter, edges)
    accepted = valid & np.isfinite(mapped).all(axis=1)
    report = {
        "input_semantics": semantics,
        "native_channels": int(len(diameter)),
        "accepted_hours": int(accepted.sum()),
        "common_cells": int(mapped.shape[1]),
        "finite_nonnegative": bool(
            np.isfinite(mapped[accepted]).all() and (mapped[accepted] >= 0).all()
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mapping_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    if not report["finite_nonnegative"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
