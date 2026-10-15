"""Build the static dashboard in site/ (served by GitHub Pages): python scripts/study/build_site.py

It reuses the outputs of the notebook: reports/results.json, paper/figures/*.png and the demo
files in app/. Everything the page needs is written to site/data.js, so it runs without a server.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from ber.decide import expected_f05_topk

SITE = ROOT / "site"
MODELS = ["LR", "LDA", "QDA", "LightGBM"]
KEEP_TOP = 8  # candidates shown per entity: top-8 of any model, every accepted candidate and every true match

(SITE / "figures").mkdir(parents=True, exist_ok=True)
for f in (ROOT / "paper" / "figures").glob("*.png"):
    shutil.copy(f, SITE / "figures" / f.name)
shutil.copy(ROOT / "paper" / "main.pdf", SITE / "paper.pdf")

res = json.loads((ROOT / "reports" / "results.json").read_text().replace("NaN", "null"))
pairs = pl.read_parquet(ROOT / "app" / "demo_pairs.parquet")
recs = pl.read_parquet(ROOT / "app" / "demo_records.parquet")
ents = pl.read_parquet(ROOT / "app" / "demo_entities.parquet")
text = {r[0]: [r[1] or "", r[2] or ""] for r in recs.select("entity_id", "business_name", "business_address").iter_rows()}

entities = []
for (s1,), g in pairs.group_by("s1_id", maintain_order=True):
    n_cand, k = g.height, {}
    for m in MODELS:
        order = np.argsort(-g[f"p_{m}"].to_numpy())
        top = set(order[:expected_f05_topk(g[f"p_{m}"].to_numpy()[order])].tolist())
        g = g.with_columns(pl.Series(f"sel_{m}", [i in top for i in range(g.height)]))
        k[m] = len(top)
    ranks = [g[f"p_{m}"].rank("ordinal", descending=True) for m in MODELS]
    keep = (pl.min_horizontal(ranks) <= KEEP_TOP) | (g["y"] == 1) | pl.any_horizontal([g[f"sel_{m}"] for m in MODELS])
    g = g.filter(keep).sort("p_LightGBM", descending=True)
    cands = [[c, *text.get(c, ["", ""]), int(y), *[round(float(x), 4) for x in ps], *[int(s) for s in sel],
              round(float(nc), 3), round(float(ac), 3), int(round(float(ts)))]
             for c, y, *rest in g.select("cand_id", "y", *[f"p_{m}" for m in MODELS], *[f"sel_{m}" for m in MODELS],
                                          "name_cos", "addr_cos", "n_tset").iter_rows()
             for ps, sel, (nc, ac, ts) in [(rest[:4], rest[4:8], rest[8:])]]
    entities.append({"id": s1, "name": text[s1][0], "addr": text[s1][1], "k": k, "cands": cands,
                     "n_cand": n_cand})

meta = dict(zip(ents["s1_id"].to_list(), zip(ents["country"].to_list(), ents["n_true"].to_list())))
for e in entities:
    e["country"], e["n_true"] = meta[e["id"]]
entities.sort(key=lambda e: (e["country"], e["name"].lower()))

payload = {"results": res, "entities": entities, "models": MODELS}
(SITE / "data.js").write_text("window.DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
                              encoding="utf-8")
(SITE / ".nojekyll").write_text("")
print(f"{len(entities)} entities, data.js {(SITE / 'data.js').stat().st_size / 1e6:.1f} MB")
