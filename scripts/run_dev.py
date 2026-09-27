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
stats = P.record_stats(recs)
recs = recs.drop("business_name", "business_address")
s1 = recs.filter(pl.col("src") == "1")

blk_name = "dev_block_smoke" if SMOKE else "dev_block"
blk_path = INTERIM / f"{blk_name}.parquet"
if not blk_path.exists():
    P.block(s1, recs.filter(pl.col("src") != "1"), cache=INTERIM / blk_name).write_parquet(blk_path)

# Stage A trains on a sample (all cheap-feature rows of every S1 would not fit in memory);
# stage B then uses every f1-f4 S1, pruned country by country right after its cheap features.
f123 = s1.filter(pl.col("split").is_in(["f1", "f2", "f3"]))["entity_id"]
f4 = s1.filter(pl.col("split") == "f4")["entity_id"]
f0_ids = s1.filter(pl.col("split") == "f0")["entity_id"]
a_ids = pl.concat([f123.sample(N_TRAIN_S1, seed=0), f4.sample(N_TRAIN_S1 // 3, seed=0)])
role = pl.concat([
    pl.DataFrame({"s1_id": f123, "role": "tr"}),
    pl.DataFrame({"s1_id": f4, "role": "va"}),
    pl.DataFrame({"s1_id": f0_ids, "role": "te"}),
]).with_columns(pl.col("s1_id").is_in(a_ids.implode()).alias("in_a"))
labels = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
countries = s1["country"].unique(maintain_order=True).to_list()


def cheap(c: str) -> pl.DataFrame:
    """Cheap features for one country (context features need every S1 of the country)."""
    ids = s1.filter(pl.col("country") == c)["entity_id"].implode()
    blk = pl.scan_parquet(blk_path).filter(pl.col("s1_id").is_in(ids)).collect()
    d = P.cheap_features(blk, stats)
    return d.join(role, on="s1_id").join(labels, on=["s1_id", "cand_id"], how="left").with_columns(pl.col("y").fill_null(0))


parts = []
for c in countries:
    parts.append(cheap(c).filter(pl.col("in_a") | (pl.col("role") == "te")))
    P.log(f"cheap features {c}: {parts[-1].height} rows kept")
df = pl.concat(parts)
del parts
n_true_f0 = gt.join(pl.DataFrame({"s1_id": f0_ids}), on="s1_id").height
blk_f0 = df.filter(pl.col("role") == "te")
block_stats = {"block_pairs_per_s1": blk_f0.height / len(f0_ids), "block_recall": blk_f0["y"].sum() / n_true_f0}
P.log(json.dumps(block_stats))

tr, va = (df.filter(pl.col("in_a") & (pl.col("role") == r)) for r in ("tr", "va"))
del df, blk_f0
model_a = P.train_lgb(tr, va, P.A_FEATS)
model_a.save_model(str(MODELS / "stage_a.txt"))

# Prune threshold: keep RECALL_KEEP of the positives that blocking found (valid sample).
pos = np.sort(P.predict(model_a, va.filter(pl.col("y") == 1), P.A_FEATS))
tau = float(pos[int((1 - RECALL_KEEP) * len(pos))])
P.log(f"tau {tau:.5f}")
del tr, va

pruned = []
for c in countries:
    pruned.append(P.prune(cheap(c), model_a, tau, MAX_CANDS))
    P.log(f"pruned {c}: {pruned[-1].height} rows")
df = pl.concat(pruned)
del pruned
tr, va, te = (df.filter(pl.col("role") == r) for r in ("tr", "va", "te"))
del df
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
