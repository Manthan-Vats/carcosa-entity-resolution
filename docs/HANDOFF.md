# HANDOFF — read this first each session

_Last updated: 2026-09-26 (session 3, in progress)_

## Goal now
Leaderboard top is 0.9906 (many teams ~0.988). **Our first submission scored 0.902 on the portal** (dev f0 estimate was 0.936). Target: ≥0.99. Deadline ≈ 2026-09-28 00:00 IST (confirm on portal).
Approach: research (papers, industry, white papers only — never other participants' repos), verify on dev, no guessing, then hand over files to upload.

## Session 3 status (read first)
- Background runs get killed under memory pressure (twice, during dev blocking). `run_dev.py` now runs in a separate PowerShell (log: `logs/run_dev_v2.log`, started ~23:20 with OLD code).
- Memory fix written, NOT yet verified or committed: `stop_words_=None` after TF-IDF fit (was pickled into 16 joblib workers); `P.block(..., cache=INTERIM/"dev_block")` writes/reuses per-country `dev_block_{country}.parquet` and frees memory between countries; run scripts compute `record_stats` then drop raw name/address columns.
- To verify once RAM is free: re-block the smoke subset and compare with the cached `dev_block_smoke.parquet` (must be identical), then commit.
- PowerShell env var syntax: `$env:PYTHONIOENCODING="utf-8"` (not `set`).

## State of the code (all under src/ber, scripts/)
- `normalize.py`: anyascii + lowercase + `&`→and; **learned Indic→Latin token dictionary** (`data/interim/translit_dict.json`, built by `scripts/build_translit.py` from dev-fold matched pairs; 1,339 name tokens, 17 address tokens; 96.4% of test Indic name tokens covered); `num_sig()` house/plot-number signature (column `num_n`).
- `data.py`: `records(split)` → cached `data/interim/{split}_records_norm.parquet` (name_n, addr_n, num_n). Caches were rebuilt with transliteration (num_n is added lazily on first load).
- `blocking.py`: per-view TF-IDF (name char-3 max_df .005, addr word, num word), joint = hstack of sqrt(w)·views, `sp_matmul_topn`.
- `pipeline.py` (v2, NOT yet run at full size): union of joint(0.3 name/0.7 addr)@30 + name-only@10 among empty-address pool + joint(0.3/0.5/0.2 num)@10 → cheap features (record stats, per-S1 and per-candidate context) → stage A LightGBM prune (keep 99.5% of blocked positives, cap 10) → full string features + **sibling (entity-centric) features** → stage B LightGBM → `decide.select` (expected-F0.5 top-k).
- `scripts/run_dev.py` (all dev S1 blocked vs dev pool; train f1-f3, early-stop f4, report f0; saves `models/`), `scripts/run_test.py` (per-country scoring, writes `outputs/*.tsv`), `scripts/blocking_union.py` (recall experiments), `--smoke` flag for a fast check.
- v1 artifacts archived in `data/interim/v1/` (models, block caches, the submitted TSVs).

## Results so far (experiments/results.csv)
| run | what | dev f0 F0.5 | cands/S1 | cand recall | portal |
|---|---|---|---|---|---|
| r001 (v1) | joint@15, prune, LGB | 0.9364 (US .953, IN .911) | 4.27 | 0.888 | **0.902** |
| v2 smoke | + empty/num views, translit, sibling feats not yet | 0.964 on smoke subset | 5.2 | 0.970 | – |
v2 blocking recall on 5k f0 sample (before prune): US 0.960, India 0.910 (v1: 0.921 / 0.854).

## Verified facts this session
- Names repeat across many businesses (12.6% of US S1 names occur >20× in pool) → address must be in blocking.
- v1 loss on f0: partial lists 0.041, empty prediction for non-singletons 0.020, singleton FP 0.003; oracle on v1 candidates 0.955 → blocking recall was the cap.
- Missed pairs (v1): empty candidate address 38%, Indic script 25%, name-or-address changed 31%, both low 6%. Remaining India misses: near-identical text where only the house number discriminates.
- Stage B probabilities are well calibrated; expected-F0.5 rule beat every global threshold and temperature on held-out halves of f0.
- Splitting digit/letter boundaries ("302college") hurt recall (fused tokens are rare, discriminative) → not done.
- Lower max_df speeds blocking but costs recall (0.002: US −6 pt) → keep 0.005.

## Why portal (0.902) < dev (0.936) — to investigate FIRST next session
1. **France** (15% of test S1, unseen in train): v1 predicted empty for 14.6% of French S1 vs 5.9% US → likely weak recall on France. Build a France proxy: leave-one-country-out (train US only → eval India, and vice versa) and inspect a random France sample of candidates (unlabeled) for normalisation gaps (Rue/R., Bd, Av., St-/Saint-, SARL/SAS/SA/EURL, accents, CEDEX, 5-digit postcodes).
2. **Denser decoys in test (D-16)**: test has 23% more pool per S1, while predicted matches per S1 on test (US 3.3, IN 3.0) look similar to dev truth (3.46) → the extra pool is probably decoys [HYP]. Dev may overstate precision. Idea: make a denser dev evaluation (e.g., score f0 S1 against dev pool while holding out part of the dev S1 so their matches stay as decoys).
3. Memory: 24 GB RAM; Background jobs get killed under memory pressure. Run one heavy job at a time; test scoring is per country.

## Next steps (in order)
1. Run `uv run python scripts/run_dev.py` (v2 incl. sibling features; ~1 h; blocking caches to `data/interim/dev_block.parquet`). Log to `logs/`, record in experiments/results.csv.
2. France / transfer analysis (above) → normalisation fixes that are country-agnostic (never hard-code countries).
3. `scripts/run_test.py` → `bash scripts/validate.sh` (+ `--check-ids`) → give user `outputs/matching_results.tsv` to upload.
4. Further levers (docs/research/06_recall_levers.md): 1:1 owner constraint post-processing, second-hop retrieval from confident matches, stage B more rounds, char-level address view for typos.

## Working agreement reminders
- Lockbox never read. Research sources: papers/industry only.
- Final zip later: `code/business_entity_resolution/` + `output/` + Documentation_template.md; candidate_pairs size matters for ranking (smaller is better).
