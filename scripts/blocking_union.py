"""Blocking recall on a fold-f0 sample: joint top-k plus name-only search among empty-address records."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.blocking import embed, joint, ranks, topn
from ber.data import records
from ber.io import load_gt_pairs
from ber.paths import ROOT

N_QUERY = int(sys.argv[1]) if len(sys.argv) > 1 else 5000

splits = pl.read_parquet(ROOT / "data" / "splits" / "train_splits.parquet")
dev = records("train").join(splits, on="entity_id").filter(pl.col("split") != "lock")
pool = dev.filter(pl.col("src") != "1")
gt = load_gt_pairs()


def pairs(q, pid, r, c, v, k):
    keep = ranks(r, v) < k
    return set(zip(q["entity_id"].to_numpy()[r[keep]], pid[c[keep]]))


for country in ["US", "India"]:
    q = dev.filter((pl.col("src") == "1") & (pl.col("split") == "f0") & (pl.col("country") == country))
    q = q.sample(min(N_QUERY, q.height), seed=0)
    pc = pool.filter(pl.col("country") == country)
    pid = pc["entity_id"].to_numpy()
    empty_idx = np.flatnonzero((pc["addr_n"] == "").to_numpy())
    truth = set(gt.join(q.select(pl.col("entity_id").alias("s1_id")), on="s1_id").iter_rows())
    print(f"{country}: pool {pc.height}, empty-address {len(empty_idx) / pc.height:.3f}", flush=True)

    mats = embed(q, pc, ["name", "addr"])
    r, c, v = topn(*joint(mats, {"name": 0.3, "addr": 0.7}), 50)
    base = {k: pairs(q, pid, r, c, v, k) for k in (15, 30, 50)}
    r, c, v = topn(mats["name"][0], mats["name"][1][empty_idx], 10)
    empty = {k: pairs(q, pid[empty_idx], r, c, v, k) for k in (5, 10)}

    def rep(label, s):
        print(f"  {label:<22} recall {len(s & truth) / len(truth):.4f}  cands/q {len(s) / q.height:.1f}", flush=True)

    for k, s in base.items():
        rep(f"joint@{k}", s)
        for ke, e in empty.items():
            rep(f"joint@{k} + empty@{ke}", s | e)
