"""Run the trained pipeline on the test split and write both submission files."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber import pipeline as P
from ber.data import records
from ber.decide import select
from ber.paths import INTERIM, OUTPUTS, ROOT

MODELS = ROOT / "models"
cfg = json.load(open(MODELS / "config.json"))
model_a = lgb.Booster(model_file=str(MODELS / "stage_a.txt"))
model_b = lgb.Booster(model_file=str(MODELS / "stage_b.txt"))

recs = records("test")
stats = P.record_stats(recs)
recs = recs.drop("business_name", "business_address")
s1 = recs.filter(pl.col("src") == "1")
blk_path = INTERIM / "test_block.parquet"
if blk_path.exists():
    blk = pl.read_parquet(blk_path)
else:
    blk = P.block(s1, recs.filter(pl.col("src") != "1"), cache=INTERIM / "test_block")
    blk.write_parquet(blk_path)
P.log(f"blocked pairs {blk.height}")


def score(pairs: pl.DataFrame) -> pl.DataFrame:
    d = P.prune(P.cheap_features(pairs, stats), model_a, cfg["tau"], cfg["max_cands"])
    d = P.full_features(d, recs)
    d = d.with_columns(pl.Series("prob", P.predict(model_b, d, P.B_FEATS)))
    return d.select("s1_id", "cand_id", "p_a", "prob")


# Countries are blocked separately, so scoring them one at a time is exact and keeps memory low.
country = s1.select(pl.col("entity_id").alias("s1_id"), "country")
blk = blk.join(country, on="s1_id")
parts = []
for c in blk["country"].unique(maintain_order=True).to_list():
    parts.append(score(blk.filter(pl.col("country") == c).drop("country")))
    P.log(f"scored {c}")
del blk
df = pl.concat(parts)
df.write_parquet(INTERIM / "test_scored.parquet")
P.log(f"candidate pairs {df.height} for {s1.height} S1")

pred = select(df.select("s1_id", "cand_id", "prob"))
cands = {k: v for k, v in df.group_by("s1_id").agg(pl.col("cand_id")).iter_rows()}

OUTPUTS.mkdir(exist_ok=True)
ids = s1["entity_id"].to_list()
with open(OUTPUTS / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for i in ids:
        f.write(f"{i}\t{','.join(pred.get(i, []))}\n")
with open(OUTPUTS / "candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for i in ids:
        f.write(f"{i}\t{','.join(cands.get(i, []))}\n")

n_match = sum(len(v) for v in pred.values())
by_country = (
    s1.select("entity_id", "country")
    .join(pl.DataFrame({"entity_id": list(pred), "n": [len(v) for v in pred.values()]}), on="entity_id", how="left")
    .with_columns(pl.col("n").fill_null(0))
    .group_by("country").agg(pl.len(), pl.col("n").mean().alias("matches_per_s1"), (pl.col("n") == 0).mean().alias("empty_rate"))
)
P.log(f"matches {n_match}; candidates per S1 {df.height / s1.height:.2f}\n{by_country}")
