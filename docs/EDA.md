# EDA Report (datasheet style)

Rules: dev folds f0–f4 only; the lockbox is never inspected. Samples are random with a fixed seed, never file heads. Every finding is tagged [CONFIRMED] (full-data computation) or [EXPLORATORY] (pattern seen in samples).

## 1. Ground-truth structure (train, full data, ids/countries only)
- [CONFIRMED] No duplicate ids in any source; all GT ids exist in the source files; no duplicate pairs.
- [CONFIRMED] Every S2/S3 id has at most one S1 owner (0 of 7,638,365 shared). Components are stars.
- [CONFIRMED] S1 and matched records always share a country (0 disagreements).
- [CONFIRMED] Unmatched pool: S2 1,340,997 (26.6%), S3 1,340,857 (25.4%); by country India 1,073,503, US 1,608,351.
- [CONFIRMED] Per S1 (non-singleton): S2 matches 0–5, S3 matches 0–6, so both S2 and S3 hold multiple copies of the same business.
- [CONFIRMED] Singletons: India 49,351, US 73,896.

## 2. Train vs test size
| split | country | S1 | pool (S2+S3) | pool per S1 |
|---|---|---|---|---|
| train | India | 883,188 | 4,133,346 | 4.68 |
| train | US | 1,323,633 | 6,186,873 | 4.67 |
| test | France | 259,452 | 1,434,993 | 5.53 |
| test | India | 809,986 | 4,717,565 | 5.82 |
| test | US | 663,106 | 3,817,031 | 5.76 |
- [CONFIRMED] Test has ~23% more pool records per S1. Cause unknown (D-16).

## 3. Pending
Profiling, open exploration, targeted checks, shift analysis, label audit, write-up.
