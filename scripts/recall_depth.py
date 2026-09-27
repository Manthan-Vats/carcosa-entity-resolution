"""Blocking recall versus depth K per view on a fold-f0 sample (v2 views), to size K."""
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.blocking import VIEWS, embed, joint, ranks, topn
from ber.data import records
from ber.io import load_gt_pairs
from ber.paths import ROOT

N_QUERY = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
KS = (10, 30, 50, 100, 200)
ZEROS = "--zeros" in sys.argv
for a in sys.argv:
    if a.startswith("--maxdf-"):  # e.g. --maxdf-addr=0.05
        view, val = a[len("--maxdf-"):].split("=")
        VIEWS[view]["max_df"] = float(val)
print({k: v["max_df"] for k, v in VIEWS.items()}, "zeros", ZEROS, flush=True)

splits = pl.read_parquet(ROOT / "data" / "splits" / "train_splits.parquet")
dev = records("train").drop("business_name", "business_address").join(splits, on="entity_id").filter(pl.col("split") != "lock")
if ZEROS:
    dev = dev.with_columns(pl.col("addr_n").str.replace_all(r"\b0+(\d)", "$1"))
pool = dev.filter(pl.col("src") != "1")
gt = load_gt_pairs()

for country in ["US", "India"]:
    q = dev.filter((pl.col("src") == "1") & (pl.col("split") == "f0") & (pl.col("country") == country)).sample(N_QUERY, seed=0)
    pc = pool.filter(pl.col("country") == country)
    pid = pc["entity_id"].to_numpy()
    qid = q["entity_id"].to_numpy()
    truth = set(gt.join(q.select(pl.col("entity_id").alias("s1_id")), on="s1_id").iter_rows())
    mats = embed(q, pc, ["name", "addr", "num"])
    found = {}
    views = {
        "joint": joint(mats, {"name": 0.3, "addr": 0.7}),
        "joint_num": joint(mats, {"name": 0.3, "addr": 0.5, "num": 0.2}),
        "name": (mats["name"][0], mats["name"][1]),
        "name_heavy": joint(mats, {"name": 0.6, "addr": 0.4}),
    }
    print(f"{country}: {len(truth)} true pairs", flush=True)
    for name, (a, b) in views.items():
        r, c, v = topn(a, b, max(KS))
        rk = ranks(r, v)
        for k in KS:
            s = set(zip(qid[r[rk < k]], pid[c[rk < k]]))
            found[(name, k)] = s
            print(f"  {name:<11}@{k:<4} recall {len(s & truth) / len(truth):.4f}", flush=True)
    for combo in [("joint", 30, "joint_num", 10), ("joint", 100, "name_heavy", 50), ("joint", 200, "name_heavy", 100), ("joint", 100, "name", 50)]:
        s = found[combo[:2]] | found[combo[2:]]
        print(f"  {combo} recall {len(s & truth) / len(truth):.4f} cands/q {len(s) / q.height:.1f}", flush=True)
    del mats, views
