"""Reading raw TSVs and cached parquet files."""
import polars as pl

from ber.paths import parquet, raw_tsv

SOURCE_SCHEMA = {
    "entity_id": pl.Utf8,
    "business_name": pl.Utf8,
    "business_address": pl.Utf8,
    "country": pl.Utf8,
}


def read_raw_source(split: str, source: str) -> pl.DataFrame:
    # quote_char=None: names/addresses may contain stray quotes; tabs are the only delimiter.
    return pl.read_csv(
        raw_tsv(split, source), separator="\t", quote_char=None,
        schema=SOURCE_SCHEMA, missing_utf8_is_empty_string=True,
    )


def read_raw_ground_truth() -> pl.DataFrame:
    return pl.read_csv(
        raw_tsv("train", "ground_truth"), separator="\t", quote_char=None,
        schema={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        missing_utf8_is_empty_string=True,
    )


def gt_to_pairs(gt: pl.DataFrame) -> pl.DataFrame:
    """Ground truth rows -> long (s1_id, match_id) pairs; singletons are dropped."""
    return (
        gt.filter(pl.col("matched_entity_ids") != "")
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.split(",").alias("match_id"),
        )
        .explode("match_id")
    )


def load_source(split: str, source: str) -> pl.DataFrame:
    return pl.read_parquet(parquet(split, source))


def load_gt_pairs() -> pl.DataFrame:
    return pl.read_parquet(parquet("train", "gt_pairs"))
