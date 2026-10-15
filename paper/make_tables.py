"""Write the LaTeX tables of the paper from reports/ (run after the notebook): python paper/make_tables.py"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
OUT = ROOT / "paper" / "tables"
OUT.mkdir(exist_ok=True)
R = json.load(open(REP / "results.json"))
MODELS = ["LR", "LDA", "QDA", "LightGBM"]


def table(name, cols, rows, caption, label, star=False, size=r"\footnotesize"):
    env = "table*" if star else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule"]
    lines += rows
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def best(vals, hi=True):
    """Bold the best value of a column."""
    b = max(vals) if hi else min(vals)
    return [rf"\textbf{{{v:.4f}}}" if v == b else f"{v:.4f}" for v in vals]


q = pd.DataFrame(R["quality"])
rows = [r"Src. & Country & Empty addr. \% & Devanagari \% & Name len. \\", r"\midrule"]
for r in q.itertuples():
    rows.append(f"S{r.src} & {r.country} & {100 * r.addr_missing:.1f} & {100 * r.indic_script:.1f} & {r.name_len_median:.0f} \\\\")
table("quality", "llrrr", rows, "Data quality of the study sample (name length: median characters)", "tab:quality")

pm = pd.DataFrame(R["pair_metrics"]).set_index("model").loc[MODELS]
cols = [("auc_roc", "AUC-ROC", True), ("pr_auc", "PR-AUC", True), ("f1", "F1", True), ("precision", "Prec.", True),
        ("recall", "Rec.", True), ("log_loss", "Log-loss", False), ("brier", "Brier", False)]
fmt = {c: best(pm[c].tolist(), hi) for c, _, hi in cols}
rows = ["Model & " + " & ".join(h for _, h, _ in cols) + r" \\", r"\midrule"]
for i, m in enumerate(MODELS):
    rows.append(f"{m} & " + " & ".join(fmt[c][i] for c, _, _ in cols) + r" \\")
table("pairs", "lrrrrrrr", rows, "Pair-level results on the lockbox (1.14M pairs, F1 threshold chosen on f4)",
      "tab:pairs", star=True)

tu = R["tuning"]
short = {
    "LR": lambda p: f"$C={p['C']:g}$, weights: {p['class_weight'] or 'none'}",
    "LDA": lambda p: f"shrinkage: {p['shrinkage'] if p['shrinkage'] is not None else 'none'}",
    "QDA": lambda p: f"reg. param $={p['reg_param']:g}$",
    "LightGBM": lambda p: f"{p['num_leaves']} leaves, lr {p['learning_rate']:.3f}",
}
rows = [r"Model & Search & Selected setting & CV AP (mean $\pm$ sd) \\", r"\midrule"]
for m in MODELS:
    s = "random" if m == "LightGBM" else "grid"
    rows.append(f"{m} & {s} & {short[m](tu[m]['best_params'])} & {tu[m]['cv_ap_mean']:.4f} $\\pm$ {tu[m]['cv_ap_std']:.4f} \\\\")
table("tuning", "llll", rows, "Hyperparameter search with 5-fold grouped cross-validation", "tab:tuning")

em = pd.DataFrame(R["entity_metrics"])
rules = ["0.5 threshold", "F1 threshold", "expected-F0.5 top-k"]
piv = em.pivot(index="model", columns="rule", values="macro_f05").loc[MODELS, rules]
top = em[em.rule == "expected-F0.5 top-k"].set_index("model").loc[MODELS]
allv = piv.values.max()
rows = [r"Model & $p\ge0.5$ & F1 thr. & Exp.-F0.5 & US & India \\", r"\midrule"]
for m in MODELS:
    v = [rf"\textbf{{{x:.4f}}}" if x == allv else f"{x:.4f}" for x in piv.loc[m]]
    rows.append(f"{m} & " + " & ".join(v) + f" & {top.loc[m, 'f05_US']:.4f} & {top.loc[m, 'f05_India']:.4f} \\\\")
rows += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{Perfect classifier on the blocked candidates: {R['entity_ceiling']:.4f}}} \\"]
table("entity", "lrrrrr", rows, r"Entity-level macro F0.5 on the lockbox (21,685 entities); US/India use the expected-F0.5 rule",
      "tab:entity")

cm = pd.DataFrame(R["count_models"]["table"])
rows = [r"Model & AIC & MAE & RMSE & Acc. (0/1/2/3+) \\", r"\midrule"]
for r in cm.itertuples():
    name = r.model.replace("Multinomial logit (0/1/2/3+)", "Multinomial logit")
    rows.append(f"{name} & {r.aic:,.0f} & {r.mae:.3f} & {r.rmse:.3f} & {r.class_acc_0123:.3f} \\\\")
table("counts", "lrrrr", rows, "Count models for the number of matches per entity (lockbox). The multinomial logit "
      "predicts capped classes, so its MAE/RMSE are on the 0--3 scale", "tab:counts")

h = pd.DataFrame(R["challenge_runs"])
desc = {"r001": "TF-IDF top-15, LightGBM prune + pair model",
        "r002": "union blocking, transliteration, sibling features",
        "r003": "r002 with stage B on more data",
        "r004": "name-word view, cross-fitted stage A"}
rows = [r"Run & Change & Dev F0.5 & Recall cap & Cands/S1 & Public LB \\", r"\midrule"]
for r in h.itertuples():
    lb = f"{r.portal:.3f}" if pd.notna(r.portal) else "--"
    rows.append(f"{r.run_id} & {desc[r.run_id]} & {r.f05:.4f} & {r.recall_ceiling:.4f} & {r.avg_candidates:.2f} & {lb} \\\\")
table("challenge", "llrrrr", rows, "Development of the full challenge system (all 1.76M dev entities, scored on fold f0)",
      "tab:challenge", star=True)
print("tables written:", sorted(p.name for p in OUT.glob("*.tex")))
