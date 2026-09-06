#!/usr/bin/env python3
"""Validate a local experiment archive and report the frozen design."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from peakst.data import load_experiment
from peakst.metrics import evaluate_spectrum
from peakst.training import run_peakst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    data = load_experiment(args.input)
    config = yaml.safe_load(args.config.read_text())
    seed = int(args.seed or config["random_seeds"][0])
    predictions = run_peakst(data, config, seed, args.device)
    ordinary_metrics = evaluate_spectrum(
        data["test_y"], predictions["ordinary_adaptation"]
    )
    peakst_metrics = evaluate_spectrum(
        data["test_y"], predictions["PeakST_tapered"]
    )
    report = {
        "status": "completed",
        "source_hours": int(len(data["source_x"])),
        "reference_hours": int(len(data["reference_x"])),
        "test_hours": int(len(data["test_x"])),
        "predictors": int(data["source_x"].shape[1]),
        "outputs": int(data["source_y"].shape[1]),
        "seed": seed,
        "device": args.device,
        "terminal_test_used_for_selection": False,
        "ordinary_metrics": ordinary_metrics,
        "peakst_metrics": peakst_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.savez_compressed(
        args.output_dir / "predictions.npz",
        datetime_recorded=data["test_time"],
        observed_cell_concentration=data["test_y"],
        **predictions,
    )
    (args.output_dir / "experiment_validation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
