# Business Entity Resolution — team carcosa

Blocking (sparse TF-IDF, several views) → stage A LightGBM pruning → stage B LightGBM pair model → per-S1 expected-F0.5 selection.
Runs on a laptop CPU (tested: 16 threads, 24 GB RAM, Windows 11, Python 3.12). No GPU, no external data, no pretrained models.

## Setup
```bash
pip install -r requirements.txt          # or: uv pip install -r requirements.txt  (Python 3.12)
```
Put the challenge data at `data/raw/student_resource/` (so that `data/raw/student_resource/dataset/{train,test}/*.tsv` exist).

## Reproduce both output files
All commands are run from this folder. On Windows set `PYTHONIOENCODING=utf-8`.

| Step | Command | What it does | Time (16 threads) |
|---|---|---|---|
| 1 | `python src/scripts/convert_to_parquet.py` | TSV → parquet cache in `data/interim/` | ~2 min |
| 2 | `python src/scripts/make_splits.py` | lockbox + 5 dev folds on the train S1 (seed 2026) → `data/splits/train_splits.parquet` (shipped) | ~1 min |
| 3 | `python src/scripts/build_translit.py` | learns the Indic→Latin token dictionary from dev-fold matched pairs → `data/interim/translit_dict.json` (shipped) | ~5 min |
| 4 | `python src/scripts/run_dev.py` | blocks all dev S1, trains the two cross-fitted stage A models and stage B, reports F0.5 on fold f0, writes `models/` (shipped) | ~5 h (blocking ~4 h) |
| 5 | `python src/scripts/block_test.py` | blocks the test split, one cache file per country | ~2.5 h |
| 6 | `python src/scripts/run_test.py` | scores test with `models/`, writes `outputs/matching_results.tsv` and `outputs/candidate_pairs.tsv` | ~1 h |
| 7 | `bash src/scripts/validate.sh --check-ids` | official validator | ~1 min |

Steps 2–4 can be skipped by using the shipped `data/splits/`, `data/interim/translit_dict.json` and `models/`; then steps 1, 5, 6 regenerate the submitted outputs.
Every heavy step caches its result per country, so an interrupted run resumes.

## Layout
```
src/ber/normalize.py   text normalisation (anyascii, learned Indic→Latin dictionary, house-number signature)
src/ber/data.py        normalised record tables (cached)
src/ber/blocking.py    TF-IDF views + sparse top-k search (sparse_dot_topn)
src/ber/pipeline.py    blocking union, stage A/B features, pruning, LightGBM training
src/ber/features.py    string, context and sibling features
src/ber/decide.py      expected-F0.5 top-k selection per S1
src/ber/metric.py      per-S1 macro F0.5 (as defined by the challenge)
src/scripts/           entry points above; src/scripts/analysis/ recall experiments used for design choices
models/                stage_a1.txt, stage_a2.txt, stage_b.txt (LightGBM, MIT), config.json (prune threshold, cap)
```

## Licences
All libraries are permissive (polars MIT, scikit-learn BSD-3, LightGBM MIT, rapidfuzz MIT, anyascii ISC, sparse_dot_topn Apache-2.0, numpy/scipy BSD).
The only models are our own LightGBM models trained on the provided training data.
