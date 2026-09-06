# Local data schema

PeakST does not distribute measurement data. A user-supplied NPZ archive must
contain the arrays below. All time arrays use UTC `datetime64[ns]` values and
must be strictly ordered within each split.

## Experiment archive

| Key | Shape | Meaning |
|---|---:|---|
| `source_x` | `(n_source, p)` | Source routine, meteorological, and calendar predictors in one common schema |
| `source_y` | `(n_source, 64)` | Source integrated number concentration per common cell, particles cm-3 |
| `source_site` | `(n_source,)` | Source site identifier used for equal-site weighting |
| `source_year` | `(n_source,)` | Source year used to define within-site-year high states |
| `reference_x` | `(n_reference, p)` | Earlier target-reference predictors |
| `reference_y` | `(n_reference, 64)` | Earlier target-reference PNSD target cells |
| `reference_time` | `(n_reference,)` | Earlier target-reference timestamps |
| `test_x` | `(n_test, p)` | Later query-period routine predictors |
| `test_y` | `(n_test, 64)` | Later PNSD targets, used only for evaluation |
| `test_time` | `(n_test,)` | Later test timestamps |
| `edges_nm` | `(65,)` | Strictly increasing common-cell edges |

`reference_time.max()` must be earlier than `test_time.min()`. The software
rejects archives that violate this condition. PNC, CPC, PNSD, event labels,
and target-derived thresholds must not appear among query-period predictors.
The common predictor schema may contain site- or year-specific missing values;
the fitted transform appends missingness indicators after median imputation.
Users should document systematically absent variables and repeat the analysis
with a stable predictor subset when missingness could identify a site.

## Native-spectrum mapping archive

| Key | Shape | Meaning |
|---|---:|---|
| `diameter_midpoint_nm` | `(m,)` | Positive, strictly increasing native diameter midpoints |
| `spectrum` | `(n, m)` | Native spectrum with explicitly declared semantics |
| `valid_hour` | `(n,)` | Upstream quality-control flag |
| `semantics` | scalar string | `dndlog10dp`, `dndln_dp`, or `bin_concentration` |
| `native_delta_log10_dp` | `(m,)`, conditional | Required for `bin_concentration` |

The archive should be generated from the least-processed provider record.
Instrument, conditioning, averaging time, units, quality-control source, and
native support must be recorded outside the archive in the user's data
provenance system.
