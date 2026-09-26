"""Pair features for (S1, candidate) rows."""
import re

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

_NUM = re.compile(r"\d+")


def _nums(s: str) -> set[str]:
    return {n.lstrip("0") or "0" for n in _NUM.findall(s)}


def _pairwise(a: list[str], b: list[str], fn) -> np.ndarray:
    return np.fromiter((fn(x, y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


def _num_feats(a: list[str], b: list[str]) -> tuple[np.ndarray, np.ndarray]:
    jac = np.full(len(a), -1.0, dtype=np.float32)
    first = np.full(len(a), -1.0, dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        nx, ny = _nums(x), _nums(y)
        if nx and ny:
            jac[i] = len(nx & ny) / len(nx | ny)
            fx, fy = _NUM.search(x), _NUM.search(y)
            first[i] = float(fx.group().lstrip("0") == fy.group().lstrip("0"))
    return jac, first


def string_features(pairs: pl.DataFrame) -> pl.DataFrame:
    """pairs needs name_n/addr_n for both sides (suffix _q for S1, _c for candidate)."""
    nq, nc = pairs["name_n_q"].to_list(), pairs["name_n_c"].to_list()
    aq, ac = pairs["addr_n_q"].to_list(), pairs["addr_n_c"].to_list()
    num_jac, num_first = _num_feats(aq, ac)
    feats = {
        "n_ratio": _pairwise(nq, nc, fuzz.ratio),
        "n_tsort": _pairwise(nq, nc, fuzz.token_sort_ratio),
        "n_tset": _pairwise(nq, nc, fuzz.token_set_ratio),
        "n_partial": _pairwise(nq, nc, fuzz.partial_ratio),
        "n_jw": _pairwise(nq, nc, JaroWinkler.similarity),
        "a_tset": _pairwise(aq, ac, fuzz.token_set_ratio),
        "a_tsort": _pairwise(aq, ac, fuzz.token_sort_ratio),
        "a_partial": _pairwise(aq, ac, fuzz.partial_ratio),
        "num_jac": num_jac,
        "num_first": num_first,
    }
    return pl.DataFrame(feats)


def group_features(df: pl.DataFrame, score_cols: list[str]) -> pl.DataFrame:
    """Context features: how a candidate compares to others for the same S1, and to other S1s."""
    exprs = []
    for c in score_cols:
        exprs += [
            (pl.col(c) - pl.col(c).max().over("s1_id")).alias(f"{c}_gap_s1"),
            pl.col(c).rank("ordinal", descending=True).over("s1_id").alias(f"{c}_rk_s1"),
            (pl.col(c) - pl.col(c).max().over("cand_id")).alias(f"{c}_gap_cand"),
            pl.col(c).rank("ordinal", descending=True).over("cand_id").alias(f"{c}_rk_cand"),
        ]
    exprs += [
        pl.len().over("s1_id").alias("n_cand_s1"),
        pl.len().over("cand_id").alias("n_s1_cand"),
    ]
    return df.with_columns(exprs)


def _sib_scores(a: list[str], b: list[str], an: list[str], bn: list[str]) -> pl.DataFrame:
    return pl.DataFrame({
        "sn": _pairwise(a, b, fuzz.token_set_ratio),
        "sa": _pairwise(an, bn, fuzz.token_set_ratio),
    })


def sibling_features(df: pl.DataFrame, recs: pl.DataFrame, n_jobs: int = 16, chunk_s1: int = 100_000) -> pl.DataFrame:
    """Entity-centric evidence: how much a candidate looks like the other strong candidates of
    the same S1 (records of one business resemble each other). Needs s1_id, cand_id, p_a.
    Features only use candidates of the same S1, so S1 chunks bound memory without changing results."""
    side = recs.select("entity_id", "name_n", "addr_n")
    ids = df["s1_id"].unique(maintain_order=True)
    aggs = [_sibling_chunk(df.filter(pl.col("s1_id").is_in(ids.slice(i, chunk_s1).implode())), side, n_jobs)
            for i in range(0, len(ids), chunk_s1)]
    return df.join(pl.concat(aggs), on=["s1_id", "cand_id"], how="left").with_columns(
        pl.col("sib_name", "sib_addr", "sib_both").fill_null(-1), pl.col("sib_n_strong").fill_null(0)
    )


def _sibling_chunk(df: pl.DataFrame, side: pl.DataFrame, n_jobs: int) -> pl.DataFrame:
    c = df.select("s1_id", "cand_id", "p_a").join(side.rename({"entity_id": "cand_id"}), on="cand_id", how="left")
    sib = (
        c.join(c.rename({"cand_id": "o_id", "p_a": "o_p", "name_n": "o_name", "addr_n": "o_addr"}), on="s1_id")
        .filter(pl.col("cand_id") != pl.col("o_id"))
    )
    step = max(1, -(-sib.height // n_jobs))
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_sib_scores)(*(sib.slice(i, step)[k].to_list() for k in ("name_n", "o_name", "addr_n", "o_addr")))
        for i in range(0, sib.height, step)
    ) or [_sib_scores([], [], [], [])]
    sib = pl.concat([sib.select("s1_id", "cand_id", "o_p"), pl.concat(parts)], how="horizontal")
    return sib.group_by("s1_id", "cand_id").agg(
        (pl.col("o_p") * pl.col("sn")).max().alias("sib_name"),
        (pl.col("o_p") * pl.col("sa")).max().alias("sib_addr"),
        (pl.col("o_p") * (pl.col("sn") + pl.col("sa")) / 2).max().alias("sib_both"),
        ((pl.col("o_p") > 0.5) & (pl.col("sn") >= 90)).sum().alias("sib_n_strong"),
    )
