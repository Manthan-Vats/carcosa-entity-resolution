"""Recall and search time of joint@30 + name_heavy@20 over a grid of per-view max_df values.
Views are embedded once with every term; a max_df setting masks the terms above it (IDF of kept
terms does not depend on max_df) and re-normalises rows."""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.blocking import VIEWS, embed, joint, topn
from ber.data import records
from ber.io import load_gt_pairs
from ber.paths import ROOT

N_QUERY = 3000
GRID = [(0.005, 0.005, 0), (0.005, 0.005, 0.01), (0.005, 0.005, 0.03), (0.005, 0.02, 0.03), (0.005, 0.05, 0.03), (0.005, 0.02, 0.1)]  # (name, addr, name words; 0 = off)
VIEWS["nword"] = dict(col="name_n", analyzer="word", token_pattern=r"\S+")
for v in VIEWS.values():
    v["max_df"] = 1.0

splits = pl.read_parquet(ROOT / "data" / "splits" / "train_splits.parquet")
dev = records("train").drop("business_name", "business_address").join(splits, on="entity_id").filter(pl.col("split") != "lock")
dev = dev.with_columns(pl.col("addr_n").str.replace_all(r"\b0+(\d)", "$1"))
pool = dev.filter(pl.col("src") != "1")
gt = load_gt_pairs()


def mask(m, keep):
    return normalize(m @ sparse.diags(keep.astype(np.float32)), copy=False).tocsr()


for country in ["US", "India"]:
    q = dev.filter((pl.col("src") == "1") & (pl.col("split") == "f0") & (pl.col("country") == country)).sample(N_QUERY, seed=2)
    pc = pool.filter(pl.col("country") == country)
    pid, qid = pc["entity_id"].to_numpy(), q["entity_id"].to_numpy()
    truth = set(gt.join(q.select(pl.col("entity_id").alias("s1_id")), on="s1_id").iter_rows())
    full = embed(q, pc, ["name", "addr", "nword"])
    df = {v: np.bincount(full[v][1].indices, minlength=full[v][1].shape[1]) / pc.height for v in full}
    for mn, ma, mw in GRID:
        cfg = [("name", mn), ("addr", ma)] + ([("nword", mw)] if mw else [])
        mats = {v: tuple(mask(m, df[v] <= t) for m in full[v]) for v, t in cfg}
        views = [({"name": 0.3, "addr": 0.7}, 30), ({"name": 0.6, "addr": 0.4}, 20)]
        if mw:
            views = [({"name": 0.15, "nword": 0.15, "addr": 0.7}, 30), ({"name": 0.3, "nword": 0.3, "addr": 0.4}, 20)]
        found, secs, over = set(), 0.0, 0.0
        for w, k in views:
            a, b = joint(mats, w)
            t0 = time.time(); topn(a[:1], b, k); over += time.time() - t0
            t0 = time.time(); r, c, _ = topn(a, b, k); secs += time.time() - t0
            found |= set(zip(qid[r], pid[c]))
        print(f"{country} name<={mn} addr<={ma} nword<={mw}: recall {len(found & truth) / len(truth):.4f} "
              f"cands/q {len(found) / N_QUERY:.1f}  {(secs - over) / N_QUERY * 1000:.2f} ms/query (overhead {over:.0f}s)", flush=True)
    del full
