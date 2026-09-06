# Reproduction notes

The accompanying study used leave-one-site-out source training, earlier
distributed target-reference days, and a later held-out test period. Model
selection used source validation or target-reference folds only.

The public interface intentionally contains no site paths or provider files.
To reproduce an analysis, users should:

1. obtain the measurements from their original providers;
2. document native PNSD semantics and conversion widths;
3. run the mapping audit before constructing model targets;
4. prepare one experiment archive following `data_schema.md`;
5. execute every prespecified seed and reference-day budget; and
6. aggregate temporal origins within sites before equal weighting of sites.

Training seeds and site replicates describe different uncertainty sources and
must not be treated as independent environmental sites.
