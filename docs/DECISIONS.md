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
| D-08 | Grey areas | Hand-written abbreviation/suffix lists allowed + documented; unlabeled test text may be used to fit TF-IDF/IDF vocab (transductive, no labels), documented | PS_ANALYSIS §5 | AGREED |
| D-09 | rtk | Installed rtk 0.48.0 via winget; `rtk init -g` NOT run (global hook on Windows unverified) | research/03 | DONE |
| D-10 | Soup CLI | MAYBE, later-stage only (LLM re-judge of borderline pairs); not the core | research/04 | AGREED |
| D-11 | Devanagari handling | Learn a Devanagari→Latin token dictionary from train matches + anyascii/skeleton fallback; IndicXlit only if needed (Linux/Kaggle) | research/04, DATA_OVERVIEW | AGREED |
| D-12 | Public repo | User checked: empty. Ignore. | user | CLOSED |
| D-13 | Decision rule | Expected-F0.5 top-k (k≥0) per S1 from calibrated probabilities | research/01, 04 | AGREED |
