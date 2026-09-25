# Research 01: ER Frameworks, Kaggle Analogues, Helper Libs, Techniques (research notes, 2026-09-25)

> Raw research findings. Nothing here is adopted until we decide it.

## Frameworks
| Tool | License | Verdict | Why |
|---|---|---|---|
| Splink (MIT, v4.0.17 Sep 2026) | MIT | BORROW-IDEAS | Unsupervised EM, weak when the signal is only 1–2 text fields; borrow the TF adjustment and clustering |
| dedupe (MIT) | MIT | BORROW-IDEAS | Its active-learning workflow isn't needed since we have labels; learned blocking predicates are a useful idea |
| recordlinkage | BSD-3 | SKIP | Stale (Py ≤3.11) |
| pyJedAI | Apache | BORROW-IDEAS | Meta-blocking, unique-mapping clustering (1:1) |
| py_entitymatching | BSD-3 | SKIP (use its feature list as a checklist) | |
| Zingg | **AGPL** + Spark | SKIP | |
| Ditto | Apache | BORROW-IDEAS | `COL name VAL …` serialization + augmentation; reimplement in modern HF |
| Sudowoodo / DeepBlocker | BSD-3 | BORROW-IDEAS | Contrastive / embedding blocking |
| Sparkly (VLDB'23) | BSD-3 | BORROW-IDEAS | **Plain top-k TF-IDF/BM25 beat 8 state-of-the-art blockers** → reproduce with sklearn + sparse_dot_topn |
| DeepMatcher, Unicorn, AnyMatch (no license file), HierGAT | — | SKIP | outdated or license issues |

## Kaggle Foursquare Location Matching 2022 (closest analogue)
Blog: https://foursquare.com/resources/blog/developer/finding-the-right-poi-match/ · 4th place repo: https://github.com/TheoViel/kaggle_foursquare · 7th place: https://future-architect.github.io/articles/20220720a/
- 1st: text + geo kNN candidates → light LightGBM to prune → LightGBM on top 40 with heavy features → XLM-R cross-encoder → post-processing.
- 4th: kNN, about 12 candidates per id (IoU ceiling 0.983) → about 120 features incl. char TF-IDF → two-level CatBoost/LGBM/XGB.
- 7th: **5 retrievers unioned**, 32 candidates per id, 97.8% recall; LightGBM **weighted against FP**; Union-Find with weak edges pruned by betweenness.
- GNN post-processing on the match graph: 0.907 → 0.946.
- **Winning pattern: union of cheap retrievers → staged GBDT on string features → transformer cross-encoder → graph post-processing.**
- Caveat: fine-tuned LLMs have mixed cross-domain results (https://arxiv.org/abs/2409.08185), which is a risk for France.

## Helper libraries
USE: rapidfuzz (MIT), jellyfish (MIT, phonetic), anyascii (ISC, covers Devanagari), cleanco (MIT, bundled suffix lists; add our own Pvt/SARL/SAS/EURL), faiss-cpu (MIT, Windows wheels), sparse_dot_topn (Apache, Windows wheels ≤3.12).
AVOID: Unidecode (GPL), deepparse (LGPL, no India), libpostal (hard to install on Windows; OSM-trained model = external-data risk), hnswlib (sdist only).

## Techniques
- Blocking: union of char 2–4-gram TF-IDF on name, TF-IDF on name+address, rare-token/exact keys, embedding + FAISS, phonetic key; within country [HYP: check that country labels agree across sources]; measure recall@k.
- Hard negatives drawn from the blocker's own top-k.
- LightGBM features: rapidfuzz ratios on raw, normalized and suffix-stripped names; address token overlap; number/PIN/ZIP agreement; TF-IDF cosine; embedding cosine; **rank features** (decisive in Foursquare).
- Cross-encoder (≤300M) blended or stacked with the GBDT.
- Post-processing: mutual-best / 1:1, S2↔S3 transitivity, graph pruning.
- Decision: **per-S1 expected-F0.5 maximization, including an "empty" option** (Ye et al. ICML 2012 https://icml.cc/2012/papers/175.pdf; Faron's Instacart implementation https://github.com/asagar60/Instacart-Market-Basket-Analysis/blob/main/f1optimization_faron.py).
