"""Build the pair dataset used in the course study (notebooks/predictive_modeling.ipynb).

The full challenge pipeline blocks 2.2M Source 1 records against 10.3M Source 2/3
records, which takes hours. For the model comparison we draw a 5% sample instead:
5% of the S1 entities (every split, lockbox included), all of their true matches,
and 5% of the S2/S3 records that match nobody. Every S2/S3 record belongs to at
most one S1 entity, so this keeps the matched/unmatched mix of the full data.

The sample goes through the same blocking as the competition pipeline. Each
candidate pair then gets the blocking scores, context features and string
similarity features. Output: data/study/pairs.parquet and data/study/entities.parquet.
"""
import sys
import time
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ber import pipeline as P
from ber.data import records
from ber.features import string_features
from ber.io import load_gt_pairs
from ber.paths import ROOT

FRAC = 0.05
SEED = 2026
OUT = ROOT / "data" / "study"
OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()

splits = pl.read_parquet(ROOT / "data" / "splits" / "train_splits.parquet")
recs = records("train").join(splits, on="entity_id")
gt = load_gt_pairs().rename({"match_id": "cand_id"})

s1_ids = recs.filter(pl.col("src") == "1")["entity_id"].sample(fraction=FRAC, seed=SEED)
matched = gt.filter(pl.col("s1_id").is_in(s1_ids.implode()))["cand_id"]
unmatched = recs.filter((pl.col("src") != "1") & ~pl.col("entity_id").is_in(gt["cand_id"].implode()))["entity_id"]
noise = unmatched.sample(fraction=FRAC, seed=SEED)
recs = recs.filter(pl.col("entity_id").is_in(pl.concat([s1_ids, matched, noise]).implode()))
gt = gt.filter(pl.col("s1_id").is_in(s1_ids.implode()))
P.log(f"sample: {len(s1_ids)} S1, {len(matched)} matched + {len(noise)} unmatched S2/S3")

stats = P.record_stats(recs)
s1 = recs.filter(pl.col("src") == "1")
blk = P.block(s1, recs.filter(pl.col("src") != "1"))
P.log(f"blocking: {blk.height} pairs, {blk.height / s1.height:.1f} per S1")

labels = gt.with_columns(pl.lit(1, pl.Int8).alias("y"))
df = (
    P.cheap_features(blk, stats)
    .join(labels, on=["s1_id", "cand_id"], how="left")
    .with_columns(pl.col("y").fill_null(0))
)

side = recs.select("entity_id", "name_n", "addr_n")
txt = (
    df.select("s1_id", "cand_id")
    .join(side.rename({"entity_id": "s1_id", "name_n": "name_n_q", "addr_n": "addr_n_q"}), on="s1_id", how="left")
    .join(side.rename({"entity_id": "cand_id", "name_n": "name_n_c", "addr_n": "addr_n_c"}), on="cand_id", how="left")
)
step = -(-txt.height // 16)
parts = Parallel(n_jobs=16)(delayed(string_features)(txt.slice(i, step)) for i in range(0, txt.height, step))
df = pl.concat([df, pl.concat(parts)], how="horizontal")

ent = (
    s1.select(pl.col("entity_id").alias("s1_id"), "country", "split", "business_name", "business_address")
    .join(gt.group_by("s1_id").len("n_true"), on="s1_id", how="left")
    .with_columns(pl.col("n_true").fill_null(0))
)
df = df.join(ent.select("s1_id", "country", "split"), on="s1_id", how="left")
df.write_parquet(OUT / "pairs.parquet")
ent.write_parquet(OUT / "entities.parquet")
# Raw text of every sampled record, for EDA and the dashboard.
recs.select("entity_id", "src", "country", "business_name", "business_address", "name_n", "addr_n").write_parquet(OUT / "records.parquet")
P.log(f"done: {df.height} pairs, positives {df['y'].sum()} of {gt.height} true pairs, {time.time() - t0:.0f}s")
