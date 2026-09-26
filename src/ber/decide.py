"""Per-S1 match selection that maximises expected F0.5 from pair probabilities."""
import numpy as np
import polars as pl


def expected_f05_topk(p: np.ndarray) -> int:
    """Best k for probabilities sorted descending. k=0 means predict nothing.

    E[F] for top-k is approximated as 1.25 * sum(p[:k]) / (k + 0.25 * sum(p));
    the empty prediction scores P(no match) = prod(1 - p).
    """
    if len(p) == 0:
        return 0
    exp_true = p.sum()
    k = np.arange(1, len(p) + 1)
    scores = 1.25 * np.cumsum(p) / (k + 0.25 * exp_true)
    best = int(np.argmax(scores))
    empty = float(np.prod(1.0 - p))
    return 0 if empty >= scores[best] else best + 1


def select(scored: pl.DataFrame, prob_col: str = "prob") -> dict[str, list[str]]:
    """scored: s1_id, cand_id, prob. Returns s1_id -> chosen candidate ids."""
    out = {}
    df = scored.sort(["s1_id", prob_col], descending=[False, True])
    for (s1,), g in df.group_by("s1_id", maintain_order=True):
        p = g[prob_col].to_numpy()
        k = expected_f05_topk(p)
        out[s1] = g["cand_id"].head(k).to_list()
    return out
