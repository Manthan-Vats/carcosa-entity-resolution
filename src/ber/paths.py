"""Central path config. Everything is relative to the project root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "student_resource"
RAW_DATASET = RAW / "dataset"
INTERIM = ROOT / "data" / "interim"
OUTPUTS = ROOT / "outputs"
VALIDATOR = RAW / "utils" / "validate_submission.py"

SPLITS = ("train", "test")
SOURCES = ("source1", "source2", "source3")


def raw_tsv(split: str, name: str) -> Path:
    return RAW_DATASET / split / f"{split}_{name}.tsv"


def parquet(split: str, name: str) -> Path:
    return INTERIM / f"{split}_{name}.parquet"
