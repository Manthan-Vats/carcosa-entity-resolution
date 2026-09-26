"""Normalised record tables, cached to data/interim."""
import polars as pl

from ber.io import load_source
from ber.normalize import add_norm, num_sig
from ber.paths import INTERIM


def records(split: str) -> pl.DataFrame:
    """All S1/S2/S3 records of a split with normalised name/address (cached)."""
    path = INTERIM / f"{split}_records_norm.parquet"
    if path.exists():
        df = pl.read_parquet(path)
        if "num_n" not in df.columns:
            df = df.with_columns(pl.col("business_address").map_elements(num_sig, return_dtype=pl.Utf8).alias("num_n"))
            df.write_parquet(path)
        return df
    df = pl.concat(
        [load_source(split, s).with_columns(pl.lit(s[-1]).alias("src")) for s in ("source1", "source2", "source3")]
    )
    df = add_norm(df)
    df.write_parquet(path)
    return df
