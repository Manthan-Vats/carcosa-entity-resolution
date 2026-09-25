"""Assign every train record to a split: lockbox or dev folds 0-4.

S1 entities are stratified by country and match count; matched S2/S3 records
follow their owner. Unmatched S2/S3 records are spread over the splits with
the same proportions, per country and source.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.io import load_gt_pairs, load_source
from ber.paths import ROOT

SEED = 2026
SPLITS = ["lock", "f0", "f1", "f2", "f3", "f4"]
WEIGHTS = np.array([0.20, 0.16, 0.16, 0.16, 0.16, 0.16])
OUT = ROOT / "data" / "splits"


def assign(df: pl.DataFrame, strata: list[str], rng: np.random.Generator) -> pl.DataFrame:
    """Stratified assignment: shuffle inside each stratum, then cut by WEIGHTS."""
    parts = []
    for _, g in df.group_by(strata, maintain_order=True):
        n = g.height
        cuts = np.floor(np.cumsum(WEIGHTS) * n).astype(int)
        labels = np.repeat(SPLITS, np.diff(np.concatenate([[0], cuts])))
        parts.append(g.select("entity_id").with_columns(split=pl.Series(rng.permutation(labels))))
    return pl.concat(parts)


def main() -> None:
    rng = np.random.default_rng(SEED)
    gt = load_gt_pairs()
    s1 = load_source("train", "source1").select("entity_id", "country")
    n_match = gt.group_by("s1_id").len().rename({"s1_id": "entity_id", "len": "n_match"})
    s1 = (
        s1.join(n_match, on="entity_id", how="left")
        .with_columns(bucket=pl.col("n_match").fill_null(0).clip(0, 6))
        .sort("entity_id")
    )
    s1_split = assign(s1, ["country", "bucket"], rng)

    matched = gt.join(s1_split, left_on="s1_id", right_on="entity_id").select(
        pl.col("match_id").alias("entity_id"), "split"
    )

    pool = pl.concat([
        load_source("train", s).select("entity_id", "country").with_columns(source=pl.lit(s))
        for s in ("source2", "source3")
    ])
    unmatched = pool.join(matched, on="entity_id", how="anti").sort("entity_id")
    unmatched_split = assign(unmatched, ["country", "source"], rng)

    out = pl.concat([s1_split, matched, unmatched_split]).sort("entity_id")
    assert out["entity_id"].n_unique() == out.height == s1.height + pool.height

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "train_splits.parquet"
    out.write_parquet(path)
    digest = hashlib.sha256(
        "\n".join(out.filter(pl.col("split") == "lock")["entity_id"].to_list()).encode()
    ).hexdigest()
    print(f"saved {path.name}: {out.height:,} rows, lockbox id sha256 {digest[:16]}")
    print(out.with_columns(kind=pl.col("entity_id").str.slice(0, 2)).pivot(
        on="kind", index="split", values="entity_id", aggregate_function="len").sort("split"))


if __name__ == "__main__":
    main()
