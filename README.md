# Predictive Modeling for Business Entity Resolution

Course project, **Predictive Analytics**
Manthan Vats (2427030314) · Peddi Reddy Srikar Reddy (2427030741)

Business records about the same company arrive from three sources with different spellings,
abbreviations, scripts (Latin and Devanagari) and missing address parts, and no shared ID.
For every Source 1 record we predict which Source 2/3 records describe the same business.
The data comes from the Amazon ML Challenge 2026 (Business Entity Resolution), in which this
repository was our team's entry (public leaderboard macro F0.5 = 0.959).

The course study recasts the problem as a set of predictive models from the syllabus:

| Syllabus topic | Where |
|---|---|
| Data preparation, missing values, outliers, EDA | notebook §1-2, `reports/data_quality.csv` |
| Feature engineering and selection | notebook §3, `src/ber/features.py`, `src/ber/blocking.py` |
| Logistic regression, coefficient estimation and inference | notebook §6, `reports/logit_coefficients.csv` |
| LDA and QDA | notebook §5, §7 |
| Poisson, negative binomial, multinomial logit (GLMs) | notebook §10, `reports/count_models.csv` |
| Cross-validation, train/test split, grid and random search | notebook §4-5 |
| F1, AUC-ROC, RMSE, MAE | notebook §7-8, §10 |
| Overfitting and underfitting | notebook §9, `paper/figures/fig_overfit.pdf` |

## Repository layout

```
notebooks/predictive_modeling.ipynb   the full study, executed with outputs (start here)
notebooks/predictive_modeling.py      the same notebook as a plain script (# %% cells)
app/app.py                            Streamlit dashboard (results + match explorer)
paper/                                IEEE two-column paper (LaTeX source, figures, PDF)
reports/                              every table the paper reports (CSV / JSON)
scripts/study/build_study_data.py     builds the 5% study dataset (sample -> blocking -> features)
src/ber/                              challenge pipeline: normalisation, blocking, features, decision rule, metric
scripts/                              challenge pipeline entry points (run_dev.py, run_test.py, ...)
experiments/results.csv               challenge development runs r001-r004
docs/                                 problem statement, design decisions, EDA notes from the challenge
```

## Running it

Python 3.12. Install dependencies:

```bash
pip install -r requirements.txt
```

**Dashboard only** (uses the small demo files shipped in `app/`):

```bash
streamlit run app/app.py
```

**Full study** (needs the challenge data; only a 1,500-entity demo sample is included, in `app/`):

1. Put the challenge data at `data/raw/student_resource/` and run `python scripts/convert_to_parquet.py`
   (or copy the cached `data/interim/train_records_norm.parquet` and `train_gt_pairs.parquet`).
2. `python scripts/study/build_study_data.py` (about 3 minutes on 16 threads) writes `data/study/`.
3. Run `notebooks/predictive_modeling.ipynb` top to bottom. It writes `reports/`, `paper/figures/` and `app/`.

**Full challenge system** (several hours on a 16-thread laptop): see `docs/package/README.md`.

## Paper

`paper/main.tex` (IEEEtran conference format) compiles on Overleaf: upload the `paper/` folder
(`main.tex`, `refs.bib`, `figures/`) and compile with pdfLaTeX + BibTeX.
