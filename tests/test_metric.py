import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.metric import f05, macro_f05


def test_ps_example():  # PS: P=2/3, R=1 -> 0.714
    assert round(f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"}), 3) == 0.714


def test_singleton_rules():
    assert f05(set(), set()) == 1.0
    assert f05({"S2-1"}, set()) == 0.0
    assert f05(set(), {"S2-1"}) == 0.0


def test_macro_counts_missing_as_empty():
    truth = {"a": [], "b": ["S2-1"]}
    assert macro_f05({}, truth) == 0.5
