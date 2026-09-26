"""Train both stages on dev folds and score the pipeline on fold f0.

Blocking runs over all dev S1 against the whole dev pool so the competition
between records matches test time. Stage models: train f1-f3, early-stop f4,
report on f0. Models are saved for scripts/run_test.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber import pipeline as P
from ber.data import records
from ber.decide import select
from ber.io import load_gt_pairs
from ber.metric import macro_f05
from ber.paths import INTERIM, ROOT

SMOKE = "--smoke" in sys.argv
MODELS = ROOT / ("models_smoke" if SMOKE else "models")
N_TRAIN_S1 = 5_000 if SMOKE else 400_000
RECALL_KEEP = 0.995
MAX_CANDS = 10

MODELS.mkdir(exist_ok=True)
splits = pl.read_parquet(ROOT / "data" / "splits" / "train_splits.parquet")
recs = records("train").join(splits, on="entity_id").filter(pl.col("split") != "lock")
gt = load_gt_pairs().rename({"match_id": "cand_id"})
if SMOKE:
    keep = recs.filter(pl.col("src") == "1").sample(40_000, seed=0)["entity_id"]
    matched = gt.filter(pl.col("s1_id").is_in(keep.implode()))["cand_id"]
    noise = recs.filter(pl.col("src") != "1").sample(100_000, seed=0)["entity_id"]
    recs = recs.filter(pl.col("entity_id").is_in(pl.concat([keep, matched, noise]).implode()))
s1 = recs.filter(pl.col("src") == "1")

blk_path = INTERIM / ("dev_block_smoke.parquet" if SMOKE else "dev_block.parquet")
if blk_path.exists():
    blk = pl.read_parquet(blk_path)
else:
    blk = P.block(s1, recs.filter(pl.col("src") != "1"))
    blk.write_parquet(blk_path)
P.log(f"blocked pairs {blk.height}")

train_ids = s1.filter(pl.col("split").is_in(["f1", "f2", "f3"]))["entity_id"].sample(N_TRAIN_S1, seed=0)
valid_ids = s1.filter(pl.col("split") == "f4")["entity_id"].sample(N_TRAIN_S1 // 3, seed=0)
f0_ids = s1.filter(pl.col("split") == "f0")["entity_id"]
role = pl.concat([
    pl.DataFrame({"s1_id": train_ids, "role": "tr"}),
    pl.DataFrame({"s1_id": valid_ids, "role": "va"}),
    pl.DataFrame({"s1_id": f0_ids, "role": "te"}),
])
labels = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
stats = P.record_stats(recs)

# Cheap features per country (context features need every S1 of the country), then keep only sampled S1.
blk = blk.join(s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
parts = []
for c in blk["country"].unique(maintain_order=True).to_list():
    d = P.cheap_features(blk.filter(pl.col("country") == c).drop("country"), stats)
    d = d.join(role, on="s1_id").join(labels, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0))
    parts.append(d)
    P.log(f"cheap features {c}: {d.height} rows kept")
df = pl.concat(parts)
del blk, parts
n_true_f0 = gt.join(pl.DataFrame({"s1_id": f0_ids}), on="s1_id").height
blk_f0 = df.filter(pl.col("role") == "te")
block_stats = {"block_pairs_per_s1": blk_f0.height / len(f0_ids), "block_recall": blk_f0["y"].sum() / n_true_f0}
P.log(json.dumps(block_stats))

tr, va, te = (df.filter(pl.col("role") == r) for r in ("tr", "va", "te"))
del df, blk_f0
model_a = P.train_lgb(tr, va, P.A_FEATS)
model_a.save_model(str(MODELS / "stage_a.txt"))

# Prune threshold: keep RECALL_KEEP of the positives that blocking found (valid fold).
pos = np.sort(P.predict(model_a, va.filter(pl.col("y") == 1), P.A_FEATS))
tau = float(pos[int((1 - RECALL_KEEP) * len(pos))])
P.log(f"tau {tau:.5f}")

tr, va, te = (P.prune(d, model_a, tau, MAX_CANDS) for d in (tr, va, te))
tr, va, te = (P.full_features(d, recs) for d in (tr, va, te))
P.log(f"stage B rows train {tr.height} valid {va.height} f0 {te.height}")
model_b = P.train_lgb(tr, va, P.B_FEATS)
model_b.save_model(str(MODELS / "stage_b.txt"))
te = te.with_columns(pl.Series("prob", P.predict(model_b, te, P.B_FEATS)))
va = va.with_columns(pl.Series("prob", P.predict(model_b, va, P.B_FEATS)))
te.select("s1_id", "cand_id", "y", "p_a", "prob").write_parquet(INTERIM / "dev_f0_scored.parquet")
va.select("s1_id", "cand_id", "y", "p_a", "prob").write_parquet(INTERIM / "dev_f4_scored.parquet")

# Evaluation on f0 over every f0 S1, singletons included.
truth = {i: [] for i in f0_ids.to_list()}
for a, b in gt.filter(pl.col("s1_id").is_in(f0_ids.implode())).iter_rows():
    truth[a].append(b)
pred = select(te.select("s1_id", "cand_id", "prob"))
stats_out = {
    "f0_s1": len(truth), **block_stats,
    "cand_pairs_per_s1": te.height / len(truth),
    "cand_recall": te["y"].sum() / n_true_f0,
    "oracle_on_cands": macro_f05(
        {k: v for k, v in te.filter(pl.col("y") == 1).group_by("s1_id").agg("cand_id").iter_rows()}, truth),
    "tau": tau,
    "f05": macro_f05(pred, truth),
}
for c in ["US", "India"]:
    ids = set(s1.filter((pl.col("split") == "f0") & (pl.col("country") == c))["entity_id"].to_list())
    stats_out[f"f05_{c}"] = macro_f05(pred, {k: v for k, v in truth.items() if k in ids})
json.dump({"tau": tau, "max_cands": MAX_CANDS}, open(MODELS / "config.json", "w"))
P.log(json.dumps(stats_out, indent=1))
