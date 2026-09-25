# Data Overview (first look, 2026-09-25, read from the zip without extracting)

Zip: `ml resource list.zip` (1.09 GB, 2.52 GB uncompressed). Contains `student_resource/` with README.md (same as the PS), Documentation_template.md, utils/validate_submission.py, dataset/{train,test}. It also has macOS junk (`__MACOSX/`, `.DS_Store`), which we skip.

## Row counts (excluding header)
| File | Rows | Size |
|---|---|---|
| train_source1 | 2,206,821 | 210 MB |
| train_source2 | 5,034,616 | 489 MB |
| train_source3 | 5,285,603 | 504 MB |
| train_ground_truth | 2,206,821 (one row per S1) | 127 MB |
| test_source1 | 1,732,544 | 175 MB |
| test_source2 | 4,887,273 | 509 MB |
| test_source3 | 5,082,316 | 506 MB |

## Country split of S1
- Train: US 1,323,633 · India 883,188
- Test: US 663,106 · India 809,986 · **France 259,452 (15% of test)**

## Matches per S1 entity (train ground truth)
| # matches | S1 entities | % |
|---|---|---|
| 0 (singleton) | 123,247 | 5.6% |
| 1 | 119,157 | 5.4% |
| 2 | 375,212 | 17.0% |
| 3 | 530,841 | 24.1% |
| 4 | 484,115 | 21.9% |
| 5 | 321,957 | 14.6% |
| 6+ | 252,292 | 11.4% |

→ Only 5.6% are singletons, and most S1 entities have 2–5 matches. **Recall matters more than we first thought.** With 4 true matches and no false positives: finding 1 → 0.625, finding 2 → 0.833, finding 3 → 0.9375.

## Noise seen in the first rows (anecdotal, only a handful of rows)
- **Devanagari script** Indian names in S2/S3: `राम मार्केटिंग प्राइवेट लिमिटेड` (= Ram Marketing Private Limited). **Cross-script matching is required.**
- Junk prefixes: `-- Holloway Peak Inc Seafood`, `<< Team Ecole`
- Website domain used as the name: `wilfordhancock.com`
- Legal suffix moved to the front + injected accent: `LLC Moncada Léarning Center`
- Words appended: `Holloway Peak Inc Seafood`
- Address reordering: `IA, Iowa City, 1064 Newton Rd, Unit 11`; uppercase + abbreviations: `105 ELM ST, MORGANTON, NC`, `63 R. DE DIEPPE`
- Empty addresses: `International South Consultants Private Ltd` with a blank address
- Full state name vs abbreviation: `Texas` vs `TX`; India addresses use `KH NO.`, district and state names
- French forms: `Sarl`, `S.A.S`, `SCI`, `Rue`, `Boulevard`, regions (Nouvelle-Aquitaine)
- IDs are random numbers (not sequential), so the ID gives no signal.
- [HYP] The data looks **synthetically generated** (injected accents in US names, templated junk), so the noise types are probably a finite, learnable set.

## Scale implications
- Test comparison space: 1.73M × ~10M, so brute force is impossible. Blocking has to be efficient (restricting to the same country helps).
- With ~20 candidates per S1 we get ~35M test pairs to featurize. That's feasible on CPU in chunks, but it rules out a heavy cross-encoder or LLM on every pair.
- Embedding ~12M test strings with a small model locally takes about 1–2 h (throughput to be measured).
- 24 GB RAM: we need Parquet/polars, chunking, and category dtypes. Training can use a subsample of S1 entities.

## Still to check (EDA phase)
One-owner rule (does each S2/S3 record map to ≤1 S1?) · do S2/S3 records exist that match no S1? · country agreement across sources · % Devanagari by source · missing-field rates · noise-type frequencies · S2 vs S3 differences · contents of validate_submission.py.
