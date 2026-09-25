"""One-off: raw TSVs -> data/interim/*.parquet (+ exploded ground-truth pairs)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ber.io import gt_to_pairs, read_raw_ground_truth, read_raw_source
from ber.paths import INTERIM, SOURCES, SPLITS, parquet

INTERIM.mkdir(parents=True, exist_ok=True)
for split in SPLITS:
    for src in SOURCES:
        df = read_raw_source(split, src)
        df.write_parquet(parquet(split, src))
        print(f"{split}_{src}: {df.height:,} rows")
gt = read_raw_ground_truth()
gt.write_parquet(parquet("train", "ground_truth"))
pairs = gt_to_pairs(gt)
pairs.write_parquet(parquet("train", "gt_pairs"))
print(f"ground_truth: {gt.height:,} rows, {pairs.height:,} pairs")
