import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.decide import expected_f05_topk


def test_empty_when_all_low():
    assert expected_f05_topk(np.array([0.1, 0.05])) == 0


def test_takes_confident_matches():
    assert expected_f05_topk(np.array([0.99, 0.98, 0.97, 0.02])) == 3


def test_no_candidates():
    assert expected_f05_topk(np.array([])) == 0
