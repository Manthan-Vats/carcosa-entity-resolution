# Research 05: EDA Methodology, Leakage/Overfitting, Profiling Tools (research notes, 2026-09-25)

> Raw research findings. Nothing here is adopted until we discuss and decide it.

## Ordered EDA checklist (CRISP-DM data understanding + Datasheets for Datasets, https://arxiv.org/abs/1803.09010)
0. Lockbox FIRST (component-aware holdout).
1. Describe: counts, schema, null/empty rates, id uniqueness, Unicode/encoding issues (polars streaming).
2. GT structure: matches per S1, S2/S3 ids shared across S1s, connected-component sizes, S2 vs S3 share.
3. Per-country slices: script, diacritics, address formats; train vs test country mix.
4. Textuality/sparsity: token/length distributions, missing rates (Primpeli & Bizer CIKM'20).
5. Corner cases: similar non-matches and dissimilar matches; ratio (FP+FN)/#matches at the best-F1 threshold of a simple similarity.
6. Duplicates within S2/S3; exact name+address duplicates.
7. Train↔test shift (make CV mirror the test structure; the training/serving-skew idea from Rules of ML).
8. Label-noise audit: hand-label a random sample of GT pairs (WDC Products found ~4% noise, κ=0.91).
9. Write-up: a datasheet-style data report before any modelling.

## Leakage/overfitting rules (Kapoor & Narayanan 2023, https://reproducible.cs.princeton.edu/)
- Don't fit preprocessing (TF-IDF/IDF, frequency encodings) across train+eval folds [L1.2]; don't choose features or thresholds on the eval fold [L1.3].
- Handle duplicates [L1.4] and non-independence [L3.2] → split by GT connected component (GroupKFold).
- Evaluate on realistic candidate ratios (1:100+), not balanced pairs (Wang et al. IJCAI'22 https://arxiv.org/abs/2205.05889).
- Leave-one-country-out as a proxy for France [L3.3].
- Don't tune thresholds and report the score on the same CV (Cawley & Talbot 2010) → nested CV or separate tuning/report splits.
- Public leaderboard: don't chase it (Ladder https://arxiv.org/abs/1502.04585; reusable holdout, Dwork 2015). Accept a change only if it beats the best by a pre-set margin; open the lockbox once.
- Most public EM benchmarks are "rather easy" (Papadakis https://arxiv.org/abs/2307.01231), so published benchmark scores don't transfer to our task.

## Tools
USE: polars built-ins (streaming); skrub TableReport on samples (BSD-3); pandera schema checks (MIT, native polars).
MAYBE: ydata-profiling (100k-row pandas sample); evidently (train-vs-test drift, Apache); cleanlab (Apache; flags suspect GT pairs once a first model gives out-of-sample probabilities); skrub deduplicate/fuzzy_join (small samples only).
SKIP: sweetviz (≤3.11), dataprep (<3.11), D-Tale (LGPL), Great Expectations (no polars), deepchecks (AGPL), whylogs (stale).
Reference only: EntityMatchingTaskProfiler (no license).

