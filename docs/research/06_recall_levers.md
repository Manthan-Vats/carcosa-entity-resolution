# Research 06: levers to close the gap to ~0.99 (2026-09-26)

Sources: research papers, industry systems, white papers only (no challenge participants' material).

## Where the v1 loss comes from (f0, 353k S1)
- F0.5 0.936; the oracle on our candidates is 0.955 → blocking recall is the main cap. Wrong decisions cost 0.019.
- Missed true pairs (sample of 1,145): empty candidate address 38%, Indic-script names 25% (India), name changed but address matches (or vice versa) 31% (joint rank ~20-30), both very different 6%.
- After transliteration + empty-address view: remaining India misses have near-identical text; the house/plot number is the only discriminating field ("809/26", "C-349", "9-02").

## Techniques and how we use them
| Lever | Source | Use here |
|---|---|---|
| Retrieve wide, prune with a learned classifier | Papadakis et al., Supervised meta-blocking (VLDB 2014); Gagliardelli et al., Generalized supervised meta-blocking (VLDB 2022) | stage A: top-k union → LightGBM prune → small candidate_pairs |
| Sparse TF-IDF top-k is a strong blocker | Paulsen et al., Sparkly (VLDB 2023) | joint name+address TF-IDF top-k |
| Several complementary blocking keys, unioned | Christen, Data Matching (2012) ch. 4; Papadakis et al. blocking survey (ACM CSUR 2020) | extra name-only search over empty-address records |
| Term-frequency adjustment: rare values carry more evidence | Fellegi-Sunter model; Splink (UK Ministry of Justice) TF adjustments | number-signature view (house/plot numbers weighted by IDF) |
| Transliteration mining from parallel pairs | Sajjad et al., A statistical model for unsupervised and semi-supervised transliteration mining (ACL 2012) | learned Indic→Latin token dictionary from aligned train pairs (96.4% test token coverage) |
| Collective / entity-centric resolution | Bhattacharya & Getoor, Collective entity resolution in relational data (TKDD 2007); Senzing entity-centric learning | pair features against the other strong candidates of the same S1 (sibling support) |
| One record, one owner | Assignment constraint (1:1 ER, e.g. Kuhn-Munkres style) | per-candidate competition features (rank of this S1 among all S1 for the candidate) |
| Decision that maximises expected F | Nan Ye et al., Optimizing F-measures: a tale of two approaches (ICML 2012); Jansche (2007) | expected-F0.5 top-k per S1 (verified best vs thresholds on held-out halves) |
