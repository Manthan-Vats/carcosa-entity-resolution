"""Block the test split only (writes data/interim/test_block_{country}.parquet); needs no models."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber import pipeline as P
from ber.data import records
from ber.paths import INTERIM

recs = records("test").drop("business_name", "business_address")
s1 = recs.filter(pl.col("src") == "1")
P.block(s1, recs.filter(pl.col("src") != "1"), cache=INTERIM / "test_block")
P.log("test blocking done")
