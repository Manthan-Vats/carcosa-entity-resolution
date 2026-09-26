# Decision Log

Status values: PROPOSED · AGREED (user approved 2026-09-25) · DONE · CLOSED · REVISIT.
Rule: nothing is built on a PROPOSED item.

| ID | Topic | Proposal | Evidence | Status |
|---|---|---|---|---|
| D-01 | Overall approach | RAD-hybrid: our own 4-stage pipeline built from proven libs/models and ideas from winning solutions; no end-to-end ER framework | research/01 (Foursquare pattern, Sparkly, framework verdicts) | AGREED |
| D-02 | Compute | Laptop is primary (CPU stages + small GPU); Kaggle via CLI for heavier fine-tunes; Colab only for interactive work | research/03 | AGREED |
| D-03 | Environment | Native Windows + uv + pinned lockfile, explicit PyTorch CUDA index, HF_HOME set, no WSL2 | research/03 | AGREED |
| D-04 | Validation | Grouped K-fold on S1 entities + leave-one-country-out (US↔India) as the France proxy; folds fixed and saved to a file | research/02, PS | AGREED |
| D-05 | First milestone | Simple end-to-end baseline (TF-IDF blocking → rapidfuzz features → LightGBM → threshold) → validated first submission, then iterate | research/01 | AGREED |
| D-06 | Model license scope | Treat the MIT/Apache + ≤8B rule as applying to EVERY model in the pipeline | PS constraint 5 | AGREED |
| D-07 | Library licenses | Prefer permissive libs (anyascii over GPL Unidecode; skip LGPL deepparse, libpostal) | research/01 | AGREED |
| D-08 | Grey areas | Hand-written abbreviation/suffix lists allowed + documented; unlabeled test text may be used to fit TF-IDF/IDF vocab (transductive, no labels), documented | PS_ANALYSIS §5 | REVISIT (research/05: IDF fitting must mirror test-time procedure inside validation) |
| D-09 | rtk | Installed rtk 0.48.0 via winget; `rtk init -g` NOT run (global hook on Windows unverified) | research/03 | DONE |
| D-10 | Soup CLI | MAYBE, later-stage only (LLM re-judge of borderline pairs); not the core | research/04 | AGREED |
| D-11 | Devanagari handling | Learn a Devanagari→Latin token dictionary from train matches + anyascii/skeleton fallback; IndicXlit only if needed (Linux/Kaggle) | research/04, DATA_OVERVIEW | AGREED |
| D-12 | Public repo | User checked: empty. Ignore. | user | CLOSED |
| D-13 | Decision rule | Expected-F0.5 top-k (k≥0) per S1 from calibrated probabilities | research/01, 04 | AGREED |
| D-14 | Lockbox + folds | 20% lockbox + 5×16% dev folds, split unit = S1 entity (proven equal to connected components: every S2/S3 id has ≤1 owner), stratified by country × match-count; unmatched S2/S3 distributed proportionally per country × source; seed 2026; `scripts/make_splits.py` → `data/splits/train_splits.parquet`; lockbox id sha256 prefix 24c6c88570ba216f. Lockbox rows are never inspected during EDA/rule writing. | GT structure analysis, research/05 | DONE |
| D-15 | EDA tooling | Don't install the K-Dense EDA skill (512 MiB cap, hides raw values, built for scientific formats); adopt its reasoning rules + report template. Skip the networkx/polars skills. Candidates pandera + skrub TableReport, to be tested on a sample before use. | read the skill source | AGREED |
| D-16 | Train/test pool ratio shift | Test pool/S1 ≈ 5.7 vs train 4.68 (all countries). Cause (more matches vs more distractors) is UNKNOWN; must be investigated before calibration/threshold design. | size analysis | OPEN |
| D-17 | Candidate set size | Portal update: a smaller candidate set per S1 ranks higher in the final evaluation. Blocking = joint TF-IDF top-15 + a cheap LightGBM prune (keep 99.5% of blocked positives, cap 10); candidate_pairs.tsv = the pruned set the pair model scores. | PROBLEM_STATEMENT update | AGREED |
| D-18 | Blocking view | Names repeat across many businesses (12.6% of US S1 names occur >20 times in the pool), so blocking scores name and address together: cos = 0.3·name(char 3-gram) + 0.7·address(word), max_df 0.005, per country. Recall@10 on 5k f0 queries: US 0.912, India 0.840. Lower max_df was faster but lost recall (0.002: US 0.853). | scripts/blocking_recall.py, logs | DONE |
