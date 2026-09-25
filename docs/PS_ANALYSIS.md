# PS Analysis: Deliverables, Rules, Levers, Open Questions

Tags: **[PS]** = stated in the problem statement · **[MATH]** = follows directly from the stated formula · **[HYP]** = hypothesis, must be verified on data before we rely on it · **[ASK]** = unclear, needs the team or organizer to decide.

## 1. Deliverables checklist

| # | Deliverable | Where | Scored? |
|---|---|---|---|
| D1 | `matching_results.tsv` (`source1_entity_id`, `matched_entity_ids`) | Portal upload + `output/` in zip | **YES, the only scored file** |
| D2 | `candidate_pairs.tsv` (`source1_entity_id`, `candidate_entity_ids`) | `output/` in zip | No, but audited (recall ceiling, reduction ratio) |
| D3 | `code/business_entity_resolution/src/`: full runnable pipeline | zip | Reproducibility audit |
| D4 | `code/business_entity_resolution/README.md`: exact steps from data → blocking → matching → output | zip | Audit |
| D5 | `code/business_entity_resolution/requirements.txt`, pinned | zip | Audit |
| D6 | `Documentation_template.md`, filled in (.md or .pdf) | zip root | Audit (methodology, blocking, model + features, other) |
| D7 | Zip named `<team_name>_submission.zip` | final submit | Required from all teams |

## 2. Hard rules (break one and you're rejected or disqualified)

1. [PS] TSV with tab separators and the exact column names. ID lists are comma-separated with no quoting and no spaces.
2. [PS] Exactly one row for **every** test S1 entity, France included. No duplicate S1 rows.
3. [PS] Lists contain only S2-/S3- IDs that exist in the test files. No S1 IDs and no duplicates inside a list.
4. [PS] Every matched ID must also appear in candidate_pairs (the validator warns if not). candidate_pairs = the **last** stage before the model scores pairs.
5. [PS] The final model must be MIT or Apache-2.0 licensed and have ≤ 8B parameters.
6. [PS] No external lookup: no ER APIs, business registries, geocoding, or internet data augmentation. The whole pipeline is reviewed.
7. [PS] `country` is an open set. Do not hard-code, filter or one-hot it to {US, India}. **France appears only in test.**
8. [PS] Run `utils/validate_submission.py` before every upload.

## 3. How the metric behaves, and the levers it gives us

- [PS] F0.5 is computed **per S1 entity** and then averaged (macro). Every S1 entity counts equally, whether it has 0 or 10 true matches.
- [PS] Singleton (no true matches): predict empty → 1.0. Predict anything → 0.0.
- [MATH] Non-singleton, predict empty → recall 0 → F0.5 = 0 (assuming the scorer treats 0/0 precision as 0) [ASK/verify with our own scorer design].
- [MATH] Precision matters much more than recall. Some examples:
  - 1 correct out of 3 true, nothing wrong: P=1, R=0.33 → **0.714**
  - 3 of 3 correct plus 1 wrong: P=0.75, R=1 → **0.789**
  - 1 correct plus 1 wrong: P=0.5, R=1 (if 1 true) → **0.556**
  → When unsure, predicting fewer, confident matches is better than predicting many.
- **Lever A: per-entity expected-F0.5 optimization** [MATH]. Once we have calibrated match probabilities for each S1 entity's candidates, we can choose the subset (including the empty set) that maximizes *expected* F0.5 for that entity, instead of using one global threshold. This is a known approach from F1-optimization work, e.g. the Instacart Kaggle competition.
- **Lever B: the empty-set decision is its own problem.** The number of singletons in the data sets how much this is worth. [HYP] We'll measure the singleton rate in train.
- **Lever C: one-to-one constraint** [HYP]. S1 is deduplicated, so each S2/S3 record probably matches **at most one** S1 entity. If the data confirms this, we resolve conflicts globally (each S2/S3 record goes to its best S1), which removes false positives cheaply.
- **Lever D: S2↔S3 consistency** [HYP]. An S2 record and an S3 record matched to the same S1 should also look similar to each other. We can use that as extra evidence.
- **Lever E: within-source duplicates** [HYP]. S2 or S3 may contain several records for the same business (the ground truth allows "many" matches). We need to check how often.
- **Public/private split** [PS]. The public leaderboard is only part of the test set, so we should trust our own validation more than public LB noise and not overfit to it.

## 4. Main technical risks

- **France is not in train (domain shift)** [PS]. French legal suffixes (SARL, SAS, SA, EURL…) and address forms (Rue, Bd/Boulevard, Av., 5-digit postcodes, CEDEX, accents) never appear in training. The features must work across countries: char n-grams, accent folding, and multilingual embeddings. Our validation should simulate an unseen country (e.g. train on US only → validate on India) so we can measure how well the pipeline transfers.
- **Blocking recall caps the final score** [PS]. Any true match missing from the candidates is lost for good.
- **Probability calibration** is needed for Lever A.

## 5. Grey areas we need to decide or ask about [ASK]

1. **Pretrained models (embeddings or small LLMs).** Most likely allowed, since the license + ≤8B rule implies models are expected. They still need to be MIT or Apache-2.0. Does the license rule cover *every* model in the pipeline (embedder + classifier) or only the "final" one? Safest reading: every model we ship.
2. **Hand-written abbreviation/suffix dictionaries** (Rd→Road, Pvt→Private, SARL, Rue…). This is domain knowledge, not an external lookup, so it's probably fine. It needs to be documented clearly. A downloaded external gazetteer or address DB is probably **not** fine.
3. **Transductive use of test data** (fitting TF-IDF vocab or IDF on test records without labels). This is common and usually allowed, but it isn't stated. We should decide and document it.
4. Submission limits per day/total, team name, and team size aren't on the page.
5. Deadline: the timer suggests ≈ 2026-09-28 00:00 IST. Needs confirming.
6. Compute: what hardware does the final pipeline have to reproduce on? Not stated.

## 6. Still unknown until we have the data

Record counts per source/country · singleton rate · match cardinality distribution · whether one S2/S3 record ever maps to 2+ S1 entities · test size and France share · what the noise looks like in practice · contents of `Documentation_template.md` and `validate_submission.py`.
