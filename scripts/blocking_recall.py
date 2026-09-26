"""Blocking recall on a sample of fold-f0 S1 queries against the full dev pool."""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.blocking import embed, joint, ranks, topn
from ber.data import records
from ber.io import load_gt_pairs
from ber.paths import ROOT

N_QUERY = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
SETTINGS = [
    {"name": 0.5, "addr": 0.5},
    {"name": 0.3, "addr": 0.7},
    {"name": 0.4, "addr": 0.3, "addr3": 0.3},
    {"name": 0.3, "addr3": 0.7},
]
KS = [3, 5, 8, 10, 15, 20, 30]

splits = pl.read_parquet(ROOT / "data" / "splits" / "train_splits.parquet")
dev = records("train").join(splits, on="entity_id").filter(pl.col("split") != "lock")
pool = dev.filter(pl.col("src") != "1")
gt = load_gt_pairs()

for country in ["US", "India"]:
    q = dev.filter((pl.col("src") == "1") & (pl.col("split") == "f0") & (pl.col("country") == country))
    q = q.sample(min(N_QUERY, q.height), seed=0)
    pc = pool.filter(pl.col("country") == country)
    truth = set(gt.join(q.select(pl.col("entity_id").alias("s1_id")), on="s1_id").iter_rows())
    t = time.time()
    mats = embed(q, pc, ["name", "addr", "addr3"])
    print(f"{country}: queries {q.height}, pool {pc.height}, true pairs {len(truth)}, embed {time.time() - t:.0f}s")
    qid, pid = q["entity_id"].to_numpy(), pc["entity_id"].to_numpy()
    for w in SETTINGS:
        t = time.time()
        r, c, v = topn(*joint(mats, w), max(KS))
        rk = ranks(r, v)
        hit = np.fromiter(((a, b) in truth for a, b in zip(qid[r], pid[c])), bool, len(r))
        res = "  ".join(f"@{k}:{hit[rk < k].sum() / len(truth):.4f}" for k in KS)
        print(f"  {w}  {res}  ({time.time() - t:.0f}s)", flush=True)
