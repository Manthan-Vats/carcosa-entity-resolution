# HANDOFF — read this first each session

_Last updated: 2026-09-25 (session 1)_

## Where we are
**Phase 0 (setup) DONE. Next: Phase 1 EDA. The plan is waiting for a go-ahead.**

## Done so far
- PS transcribed → `docs/PROBLEM_STATEMENT.md`; analysis → `docs/PS_ANALYSIS.md`.
- Research (4 agents) → `docs/research/01..04_*.md`. Decisions D-01..D-13 → `docs/DECISIONS.md` (all AGREED/DONE/CLOSED).
- Data overview → `docs/DATA_OVERVIEW.md`. Key numbers: train S1 2.21M / S2 5.03M / S3 5.29M; test S1 1.73M (US 663k, India 810k, France 259k) / S2 4.89M / S3 5.08M; 7.64M GT pairs; singletons 5.6%, most S1 have 2–5 matches.
- Data extracted to `data/raw/student_resource/`; parquet cache in `data/interim/` (1.1 GB). Zip kept at root.
- uv env (Py 3.12; polars 1.44.2, lightgbm 4.7.0, sparse_dot_topn 1.2.0, faiss-cpu 1.15.1, rapidfuzz, jellyfish, anyascii, sklearn). Torch and sentence-transformers are NOT installed in the venv yet (add when we reach the embedding stage; use the CUDA index).
- `src/ber/{paths,io,metric}.py`; metric tests pass (`tests/test_metric.py`). `scripts/validate.sh` wraps the official validator. `experiments/results.csv` log header created.
- rtk 0.48.0 installed (winget); `rtk init -g` not run.
- git initialized (see `git log`).

## Agreed pipeline (D-01)
Normalize (anyascii, legal-form/abbr maps, learned Devanagari→Latin token dict) → blocking union (per-country char TF-IDF top-k via sparse_dot_topn + key blocks + later embeddings/FAISS) → LightGBM on pair features (rapidfuzz, TF-IDF cosine, rank, numbers/PIN) → [optional cross-encoder on top candidates] → per-S1 expected-F0.5 top-k (k≥0), 1:1 constraint if the data supports it. Validation: grouped by S1 + leave-one-country-out.

## Open questions
1. Approve the Phase 1 EDA plan (see the last chat message / below).
2. Turn on Windows Developer Mode (so the HF cache can use symlinks; saves disk)?
3. Team work split among the 3 members.

## Phase 1 EDA plan (proposed)
1. One-owner check: can an S2/S3 record belong to more than one S1?
2. Share of S2/S3 records that match no S1 at all (distractors).
3. Do the countries of matched pairs agree? (Is blocking within a country safe?)
4. Devanagari share by source × country; missing-field rates.
5. Noise catalogue from ~300 sampled matched pairs per country (name + address noise types, S2 vs S3 differences).
6. How hard is it: exact and near-exact match rates after simple normalization.
7. A small France sample from test (unlabeled) to see which formats to expect.
