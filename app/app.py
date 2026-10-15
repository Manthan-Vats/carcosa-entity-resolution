"""Streamlit dashboard for the course study.  Run from the repo root:  streamlit run app/app.py"""
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl
import streamlit as st

sys_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(sys_root / "src"))
from ber.decide import expected_f05_topk  # noqa: E402

APP = sys_root / "app"
REP = sys_root / "reports"
FIG = sys_root / "paper" / "figures"
MODELS = ["LR", "LDA", "QDA", "LightGBM"]
NAMES = {"LR": "Logistic regression", "LDA": "LDA", "QDA": "QDA", "LightGBM": "LightGBM"}

st.set_page_config(page_title="Entity Resolution Study", page_icon="🔗", layout="wide")


@st.cache_data
def load():
    res = json.load(open(REP / "results.json"))
    pairs = pl.read_parquet(APP / "demo_pairs.parquet")
    recs = pl.read_parquet(APP / "demo_records.parquet")
    ents = pl.read_parquet(APP / "demo_entities.parquet")
    return res, pairs, recs, ents


res, pairs, recs, ents = load()
cfg = json.load(open(APP / "models" / "config.json"))

st.title("Business Entity Resolution: Predictive Modeling Study")
st.caption("Manthan Vats (2427030314) · Peddi Reddy Srikar Reddy (2427030741) · Predictive Analytics course project · "
           "data: Amazon ML Challenge 2026 training set, 5% sample, lockbox split as test set")

tab_over, tab_eda, tab_models, tab_count, tab_demo = st.tabs(
    ["Overview", "Data & EDA", "Model comparison", "Count models", "Match explorer"])

with tab_over:
    d = res["data"]
    best = max(res["entity_metrics"], key=lambda r: r["macro_f05"])
    c = st.columns(4)
    c[0].metric("Records in sample", f"{d['records']:,}")
    c[1].metric("Source 1 entities", f"{d['s1']:,}")
    c[2].metric("Candidate pairs", f"{d['pairs']:,}", f"{d['pairs_per_s1']:.1f} per entity", delta_color="off")
    c[3].metric("Best macro F0.5 (lockbox)", f"{best['macro_f05']:.4f}", f"{best['model']}, {best['rule']}", delta_color="off")
    st.markdown(
        """
**Task.** Each Source 1 record must be linked to every Source 2/3 record that describes the same business.
Names and addresses differ through abbreviations, typos, word order, Devanagari script and missing parts.

**Approach.**
1. *Blocking*: TF-IDF cosine search on names and addresses narrows ~10M records to a few dozen candidates per entity.
2. *Features*: 43 similarity, context and record features per candidate pair.
3. *Classification*: logistic regression, LDA, QDA and LightGBM predict P(match).
4. *Decision*: per entity, the subset of candidates that maximises expected F0.5.
5. *Count models*: Poisson / negative binomial / multinomial logit predict how many matches an entity has.
""")
    st.subheader("Full challenge system (all data)")
    st.dataframe(pd.DataFrame(res["challenge_runs"]), hide_index=True, width="stretch")

with tab_eda:
    st.subheader("Data quality by source and country")
    st.dataframe(pd.DataFrame(res["quality"]), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.image(str(FIG / "fig_match_counts.png"), caption="True matches per Source 1 entity")
    b.image(str(FIG / "fig_feature_dist.png"), caption="Similarity features: matches vs non-matches")
    a.image(str(FIG / "fig_corr.png"), caption="Feature correlation (used for feature selection)")
    fs = res["feature_selection"]
    b.markdown(f"**Feature selection.** {fs['all']} features; {fs['all'] - fs['kept']} dropped for the linear "
               f"and discriminant models because |r| > 0.95 with a kept feature: `{', '.join(fs['dropped'])}`")

with tab_models:
    st.subheader("Pair-level metrics on the lockbox")
    pm = pd.DataFrame(res["pair_metrics"])
    st.dataframe(pm[["model", "auc_roc", "pr_auc", "f1", "precision", "recall", "log_loss", "brier", "threshold", "train_auc"]]
                 .round(4), hide_index=True, width="stretch")
    st.image(str(FIG / "fig_roc_pr.png"), caption="ROC (log x-axis) and precision-recall curves")
    st.subheader("Entity-level macro F0.5 (challenge metric)")
    em = pd.DataFrame(res["entity_metrics"])
    st.dataframe(em.pivot(index="model", columns="rule", values="macro_f05").round(4), width="stretch")
    st.caption(f"Ceiling with a perfect classifier on the blocked candidates: {res['entity_ceiling']:.4f}")
    st.subheader("Hyperparameter tuning (5-fold grouped CV, average precision)")
    st.dataframe(pd.DataFrame([{"model": k, "cv_ap": v["cv_ap_mean"], "std": v["cv_ap_std"], "best": str(v["best_params"])}
                               for k, v in res["tuning"].items()]).round(4), hide_index=True, width="stretch")
    a, b = st.columns(2)
    a.image(str(FIG / "fig_logit_coef.png"), caption="Logistic regression coefficients (standardised)")
    b.image(str(FIG / "fig_overfit.png"), caption="Overfitting: rounds and capacity")

with tab_count:
    cm = res["count_models"]
    st.markdown(f"Match count mean {res['count_stats']['mean']:.3f}, variance {res['count_stats']['var']:.3f}. "
                f"Poisson Pearson dispersion **{cm['pearson_dispersion']:.3f}**, negative binomial alpha "
                f"**{cm['nb_alpha']:.4f}**.")
    st.dataframe(pd.DataFrame(cm["table"]).round(4), hide_index=True, width="stretch")
    cg = res["count_guided_f05"]
    st.markdown(f"Selecting the top round(mu) candidates by {cg['model']} gives macro F0.5 **{cg['macro_f05']:.4f}**.")

with tab_demo:
    st.markdown("Pick a held-out Source 1 entity (lockbox, never used for training). The table shows its "
                "candidates from blocking, each model's match probability and the true label.")
    left, right = st.columns([1, 2])
    country = left.radio("Country", ["US", "India"], horizontal=True)
    model = left.selectbox("Model", MODELS, index=3, format_func=NAMES.get)
    only_matched = left.checkbox("Only entities with at least one true match", value=True)
    pool = ents.filter(pl.col("country") == country)
    if only_matched:
        pool = pool.filter(pl.col("n_true") > 0)
    ids = pool["s1_id"].to_list()
    name = dict(zip(recs["entity_id"].to_list(), recs["business_name"].to_list()))
    addr = dict(zip(recs["entity_id"].to_list(), recs["business_address"].to_list()))
    pick = left.selectbox("Entity", ids, format_func=lambda i: f"{name.get(i, i)[:60]}")
    left.markdown(f"**{name.get(pick)}**  \n{addr.get(pick) or '*(no address)*'}")
    n_true = int(pool.filter(pl.col("s1_id") == pick)["n_true"][0])
    left.caption(f"{n_true} true match(es) in the data")

    g = pairs.filter(pl.col("s1_id") == pick).sort(f"p_{model}", descending=True)
    k = expected_f05_topk(g[f"p_{model}"].to_numpy())
    view = pd.DataFrame({
        "selected": [i < k for i in range(g.height)],
        "candidate": g["cand_id"].to_list(),
        "name": [name.get(c) for c in g["cand_id"].to_list()],
        "address": [addr.get(c) for c in g["cand_id"].to_list()],
        "P(match)": g[f"p_{model}"].round(4).to_list(),
        "true match": g["y"].cast(pl.Boolean).to_list(),
        **{f"p {m}": g[f"p_{m}"].round(3).to_list() for m in MODELS if m != model},
        "name cos": g["name_cos"].round(3).to_list(),
        "addr cos": g["addr_cos"].round(3).to_list(),
        "name token-set": g["n_tset"].round(0).to_list(),
    })
    tp = int((view["selected"] & view["true match"]).sum())
    right.markdown(f"Expected-F0.5 rule selects **{k}** of {g.height} candidates · correct: **{tp}** · "
                   f"missed: **{n_true - tp}**")
    right.dataframe(view, hide_index=True, width="stretch", height=520)
