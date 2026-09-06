# Reproduction notes

The accompanying study used leave-one-site-out source training, earlier
distributed target-reference days, and a later held-out test period. Source
hyperparameters were selected on source-site validation periods. Adaptation
settings and allocation controls were selected in chronologically blocked
target-reference folds. The later test period was evaluated once after these
choices were fixed.

The public interface intentionally contains no site paths or provider files.
To reproduce an analysis, users should:

1. obtain the measurements from their original providers;
2. document native PNSD semantics and conversion widths;
3. run the mapping audit before constructing model targets;
4. prepare one experiment archive following `data_schema.md`;
5. select source configurations without target-site test observations;
6. select reference-period controls from blocked out-of-fold predictions;
7. refit the selected procedure on all eligible target-reference hours;
8. evaluate all methods on identical later test hours; and
9. average temporal origins within sites, sites within seeds, and seeds last.

Training seeds and site replicates describe different uncertainty sources and
must not be treated as independent environmental sites.

The example command runs a chosen configuration. Reconstructing the complete
paper analysis requires looping over source-validation candidates, target
sites, eligible temporal origins, reference-day budgets, and all reported
seeds. `peakst.selection` contains the blocked-fold and reference-only control
rules; `peakst.reporting` implements the manuscript aggregation order.
