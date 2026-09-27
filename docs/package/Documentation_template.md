# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** carcosa  
**Team Members:** {{MEMBERS}}  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary
We match every Source 1 business to its Source 2/3 records with a four-stage pipeline that runs on a laptop CPU: a union of sparse TF-IDF searches over several text views (blocking), a LightGBM pruner that keeps a small candidate set, a LightGBM pair model with string, context and "sibling" (entity-centric) features, and a per-entity decision rule that maximises expected F0.5. The largest gains came from measuring where true matches are lost: tuning which tokens the blocking keeps (a name-word view with a looser frequency cut-off) raised blocking recall on our held-out fold from 0.940 to 0.978, and a learned Indic→Latin token dictionary handles names written in Indian scripts. Dev F0.5 (fold f0, 353k S1): {{DEV_F05}}; public leaderboard: {{PORTAL}}.

---

## 2. Methodology

### 2.1 Problem Analysis
Findings from EDA on the training data (all analysis on dev folds; a 20 % lockbox was never inspected):
- **Every S2/S3 record belongs to at most one S1** (checked on the ground truth), so the S1 entity is the unit for splitting: 20 % lockbox + five 16 % folds, stratified by country × number of matches, seed 2026.
- **Names repeat heavily**: 12.6 % of US S1 names occur more than 20 times in the pool (chains, generic names), so the name alone cannot identify a business; name and address must be scored together.
- **Noise in the S2/S3 copies**: truncated addresses (often only "house number, city, state code"), state names vs codes (Karnataka / KA), zero-padded numbers (00604 vs 604), extra or dropped words in names (Services, Center, Private Limited), typos (Csenter, 5hakti), and ~3 % empty addresses.
- **Indian-script records** (Devanagari, Kannada, Gujarati, Tamil, …) are word-by-word transliterations of the Latin S1 name.
- **Test differs from train**: France appears only in test (15 % of test S1), and test has ~5.7 pool records per S1 vs 4.7 in train.
- We checked for leakage: record ids and file row order carry no match information (correlations ≈ 0), so nothing but the text is used.

### 2.2 Solution Strategy
**Approach Type:** Blocking + two-stage classifier (retrieve wide, prune with a learned model, then match) — supervised meta-blocking followed by a pairwise model and an entity-level decision.  
**Core Innovation:** measurement-driven blocking design (recall vs cost measured for every view and frequency cut-off), a name-word view that recovers name evidence cheaply, cross-fitted stage A scores feeding the pair model, entity-centric sibling features, and an expected-F0.5 top-k rule instead of a global threshold.

Country is used only as a partition (records are compared within their own country); no rule is specific to any country, so France is handled by the same pipeline.

---

## 3. Candidate Generation (Blocking)
Text normalisation (all countries): anyascii transliteration + lower-casing + `&`→and; Indian-script tokens are first mapped with a dictionary learned from matched training pairs (1,339 name tokens; 96.4 % of test Indian-script name tokens covered); leading zeros stripped from numbers; a house/plot-number signature (digit groups and joined multi-part numbers, e.g. "9-02" → "9 2 92").

- **Blocking keys used:** per-country sparse TF-IDF views (IDF fitted on the unlabelled records of the split, sublinear tf):
  - name character 3-grams (typo-robust), max_df 0.005
  - name words, max_df 0.03 (whole words are rare, so frequent-but-useful name tokens can be kept cheaply)
  - address words, max_df 0.005; house-number signature words, max_df 0.005
  
  Four exact top-k searches (sparse_dot_topn) are unioned: joint name+address score top-30; name-heavy score top-20; name+address+number score top-10; and name-only top-10 among pool records with an empty address (which the joint score cannot rank).
- **Candidate pairs generated:** blocking ≈ 55 per S1 before pruning; the stage A LightGBM pruner keeps candidates with probability ≥ τ (τ set to keep 99.5 % of the blocked true matches on a validation fold) and at most 10 per S1. `candidate_pairs.tsv` contains this pruned set: {{CANDS_PER_S1}} candidates per S1 on test ({{CANDS_TOTAL}} pairs).
- **How you ensured true matches were not lost:** we measured blocking recall on held-out S1 for every design choice (logs of recall vs depth and vs frequency cut-off). Findings: a too-strict frequency cut-off on name 3-grams deleted most name evidence; adding the name-word view raised recall at ~32 candidates from 0.954 → 0.976 (US) and 0.901 → 0.955 (India) on samples, at 1.6–1.9× search cost; denser settings reached 0.98+ but cost 9–14× and did not fit the time budget. On the full f0 fold (353k S1), blocking recall rose from 0.940 to 0.978. The pruner loses < 1 % of the blocked true matches.

---

## 4. Matching Model

**Features used (stage B, 52; stage A uses the 32 cheap ones):**
- Name features: TF-IDF cosines (char 3-gram, word), rapidfuzz ratio / token-sort / token-set / partial ratio, Jaro-Winkler.
- Address features: TF-IDF word cosine, token-set / token-sort / partial ratio, house-number cosine, Jaccard of number sets, first-number equality.
- Context features: for each score, the gap to the best candidate of the same S1 and the rank within the S1; the gap to the best S1 of the same candidate and the rank within the candidate (one record, one owner); number of candidates per S1 and per candidate; search ranks in each blocking view; stage A probability and its context.
- Other: entity-centric "sibling" features — how strongly a candidate resembles the other probable candidates of the same S1 (records of one business resemble each other); lengths, empty address, Indian-script flag, source (S2/S3).

**Model type:** LightGBM (binary, 127 leaves, early stopping on a separate fold). Stage A is cross-fitted: two models on disjoint halves, each half scored out-of-fold, other S1 get the average, so stage B trains on stage A scores that behave as they do at test time.  
**Threshold selection method:** no global threshold. For each S1 the candidates are sorted by probability and the number to predict (k ≥ 0) maximises the expected F0.5 given the calibrated probabilities; predicting nothing is chosen when P(no match) is higher. On held-out halves of the validation fold this beat every global threshold.

---

## 5. Results & Error Analysis

| Run | Change | Dev F0.5 (f0, 353k S1) | Blocking recall | Public LB |
|---|---|---|---|---|
| r001 | joint TF-IDF top-15 + prune + LightGBM | 0.936 | 0.893 | 0.902 |
| r002 | + empty-address and number searches, Indic dictionary, sibling features | 0.955 | 0.940 | 0.927 |
| r004 | + name-word view, name-heavy search, cross-fitted stage A | {{DEV_F05}} | 0.978 | {{PORTAL}} |

- **F_0.5 Score (macro):** {{DEV_F05}} on fold f0 (US {{DEV_US}}, India {{DEV_IN}}).
- **Common false positives (wrong merges):** different businesses at the same address with similar generic names (e.g. two "… Services" at one building); chain branches with identical names and nearby numbers. Precision on dev is ≈ 0.99, so these are rare.
- **Common false negatives (missed matches):** copies whose address is reduced to "number, city, state code" and whose name is in an Indian script or heavily changed (extra/dropped words plus typos); copies with an empty address and a common name. Loss decomposition for r002: 63 % of missed pairs were never retrieved by blocking (addressed in r004), 32 % were retrieved but rejected by the pair model.
- **France:** no labels exist; aggregate checks show French top candidates have the highest address similarity of all countries, and the French candidates the model rejects have very different names and numbers, consistent with genuinely unmatched S1 rather than a normalisation failure.

---

## 6. Conclusion
A classic blocking + learned matching pipeline, driven by measuring where true matches are lost, reached {{PORTAL}} on the public leaderboard on a laptop CPU with no external data. The decisive lessons were that blocking recall caps everything, that token-frequency cut-offs must be tuned per view (names need a word view), and that the decision rule should optimise the metric per entity. Next steps would be a learned dense retriever for the hardest truncated/transliterated copies and a denser address view on more compute.

---

## Appendix

### A. Code Artefacts
`code/business_entity_resolution/` — see its `README.md` for exact commands.
- Entry points: `src/scripts/convert_to_parquet.py` → `make_splits.py` → `build_translit.py` → `run_dev.py` (trains, writes `models/`) → `block_test.py` → `run_test.py` (writes `outputs/matching_results.tsv` and `outputs/candidate_pairs.tsv`) → `validate.sh`.
- Library code in `src/ber/` (normalise, block, features, pipeline, decide, metric). Trained models are shipped in `models/`.

### B. Additional Results
Blocking recall on held-out samples (joint top-30 + name-heavy top-20; ~40 candidates per query):

| Setting | US | India | Search cost |
|---|---|---|---|
| name 3-gram + address, max_df 0.005 | 0.954 | 0.901 | 1× |
| + name words max_df 0.03 (used) | 0.976 | 0.955 | 1.6–1.9× |
| + address max_df 0.02 | 0.978 | 0.968 | 2.2–3.7× |
| name/address max_df 0.05 | 0.982 | 0.978 | 9–14× |

Candidate-set size vs F0.5 (r002, dev f0): raising the stage A threshold from 0.013 to 0.03 cut candidates per S1 by 14 % for −0.0005 F0.5.
