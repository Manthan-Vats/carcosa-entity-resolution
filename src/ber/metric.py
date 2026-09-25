"""Exact competition metric: per-S1 F0.5, macro-averaged over ALL S1 entities.

Conventions (from the PS): empty truth + empty pred = 1.0; empty truth + any pred = 0.0;
non-empty truth + empty pred = 0.0 (recall 0).
"""
from collections.abc import Iterable, Mapping

BETA2 = 0.25


def f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(pred: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]) -> float:
    """Average over every S1 id in `truth`; S1 ids missing from `pred` count as empty."""
    total = 0.0
    for s1, t in truth.items():
        total += f05(set(pred.get(s1, ())), set(t))
    return total / len(truth)
