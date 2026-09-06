# PeakST

PeakST is a reproducible implementation of peak-sensitive cross-site transfer
for particle number size distributions (PNSDs). It learns an ordinary
full-spectrum predictor and a high-concentration predictor, then uses a
source-trained hourly score to allocate the correction at small particle
diameters. The repository also provides the preprocessing audits and endpoint
definitions used in the accompanying study.

This repository contains code only. Measurement data, trained weights,
predictions, manuscript files, and study figures are not distributed.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/bubaizhanshen/PeakST.git
cd PeakST
python -m pip install -e .
```

Install the development dependencies to run the tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Data interface

Users obtain PNSD and routine-monitoring observations from their original
providers and convert them to the documented local NPZ interface. Required
arrays, units, time ordering, and representation semantics are defined in
[`docs/data_schema.md`](docs/data_schema.md). No input file is uploaded by the
pipeline.

## Quick start

First audit the PNSD representation and mapping:

```bash
python scripts/audit_mapping.py \
  --input /path/to/local_site.npz \
  --config configs/example.yaml \
  --output-dir /path/to/local_audit
```

Then run one leave-target-site experiment:

```bash
python scripts/run_experiment.py \
  --input /path/to/local_experiment.npz \
  --config configs/example.yaml \
  --output-dir /path/to/local_results
```

The experiment command never searches for data outside the explicitly supplied
input path. Output directories are user-selected and ignored by Git.

## Repository layout

- `src/peakst/`: representation checks, model components, transfer operations,
  and evaluation metrics
- `scripts/`: command-line entry points
- `configs/`: a documented example configuration without site-specific paths
- `tests/`: synthetic unit tests; no measurement records
- `docs/`: data contract and reproduction notes

## Reproducibility boundaries

The software enforces three rules used in the study:

1. target-reference observations precede terminal test observations;
2. query-period PNC or PNSD is not accepted as a model input; and
3. all fit or selection operations use source or target-reference data only.

The 64 output cells are a common numerical representation, not 64 independent
instrument channels. The numerical return above 25 nm is an output
construction rather than an atmospheric constraint.

## Citation

If you use this software, cite the repository metadata in
[`CITATION.cff`](CITATION.cff). The associated manuscript citation will be
added after publication.

## License

The code is released under the MIT License. Data obtained from external
providers remain subject to their original terms.
