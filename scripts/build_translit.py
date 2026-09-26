"""Learn an Indic-script -> Latin token dictionary from matched train pairs (dev folds only).

Names: an Indic-script match is a word-by-word transliteration of the S1 name, so
tokens are aligned by position. Addresses: each Indic token is mapped to the S1
address token it co-occurs with most often.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl
from anyascii import anyascii
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ber.data import records
from ber.io import load_gt_pairs
from ber.normalize import INDIC, norm_text
from ber.paths import INTERIM, ROOT

MIN_COUNT = 2
MIN_SHARE = 0.5

splits = pl.read_parquet(ROOT / "data" / "splits" / "train_splits.parquet")
dev_ids = splits.filter(pl.col("split") != "lock")["entity_id"]
recs = records("train").filter(pl.col("entity_id").is_in(dev_ids.implode()))
side = recs.select("entity_id", "business_name", "business_address")
pairs = (
    load_gt_pairs()
    .join(side.rename({"entity_id": "s1_id", "business_name": "qn", "business_address": "qa"}), on="s1_id")
    .join(side.rename({"entity_id": "match_id", "business_name": "cn", "business_address": "ca"}), on="match_id")
)
has_indic = INDIC.pattern

name_counts: dict[str, Counter] = defaultdict(Counter)
for q, c in pairs.filter(pl.col("cn").str.contains(has_indic)).select("qn", "cn").iter_rows():
    qt, ct = q.split(), c.split()
    if len(qt) != len(ct):
        continue
    for a, b in zip(ct, qt):
        if INDIC.search(a):
            name_counts[a.strip(".,()[]")][norm_text(b)] += 1

addr_counts: dict[str, Counter] = defaultdict(Counter)
for q, c in pairs.filter(pl.col("ca").str.contains(has_indic)).select("qa", "ca").iter_rows():
    q_tokens = set(norm_text(q).split())
    for part in re.split(r"[,\s]+", c):
        if INDIC.search(part):
            addr_counts[part.strip(".,()[]")].update(q_tokens)


def best(counts: dict[str, Counter], denom: dict[str, int] | None = None) -> dict[str, str]:
    """Most frequent target; near-ties (>= 90% of the top count) go to the one closest to anyascii(tok)."""
    out = {}
    for tok, cnt in counts.items():
        top = cnt.most_common(1)[0][1]
        rough = anyascii(tok).lower()
        val, n = max(((v, c) for v, c in cnt.items() if c >= 0.9 * top), key=lambda x: fuzz.ratio(x[0], rough))
        total = denom[tok] if denom else sum(cnt.values())
        if val and n >= MIN_COUNT and n / total >= MIN_SHARE:
            out[tok] = val
    return out


# For addresses the denominator is how often the Indic token appears, not the sum of co-occurrences.
addr_seen = Counter()
for c in pairs.filter(pl.col("ca").str.contains(has_indic))["ca"].to_list():
    addr_seen.update({p.strip(".,()[]") for p in re.split(r"[,\s]+", c) if INDIC.search(p)})

vocab = {"name": best(name_counts), "addr": best(addr_counts, addr_seen)}
out = INTERIM / "translit_dict.json"
json.dump(vocab, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print({k: len(v) for k, v in vocab.items()})
