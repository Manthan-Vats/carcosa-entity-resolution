"""Text normalisation shared by blocking and features."""
import json
import re
from functools import cache

import polars as pl
from anyascii import anyascii

from ber.paths import INTERIM

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Devanagari through Malayalam blocks (Hindi, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam).
INDIC = re.compile(r"[ऀ-ൿ]")
_SPLIT = re.compile(r"([\s,]+)")
_STRIP = ".,()[]\"'"
TRANSLIT = INTERIM / "translit_dict.json"
_NUMRUN = re.compile(r"\d+(?:[/\-.]\d+)*")
_DIGITS = re.compile(r"\d+")


@cache
def _translit(kind: str) -> dict[str, str]:
    if not TRANSLIT.exists():
        return {}
    return json.load(open(TRANSLIT, encoding="utf-8"))[kind]


def _map_indic(s: str, kind: str) -> str:
    d = _translit(kind)
    return "".join(d.get(t.strip(_STRIP), t) if INDIC.search(t) else t for t in _SPLIT.split(s))


def norm_text(s: str | None, kind: str = "name") -> str:
    if not s:
        return ""
    if INDIC.search(s):
        s = _map_indic(s, kind)
    s = anyascii(s).lower().replace("&", " and ").replace("+", " and ")
    return _NON_ALNUM.sub(" ", s).strip()


def num_sig(s: str | None) -> str:
    """House/plot number signature: digit groups without leading zeros, plus the joined form
    of multi-part numbers, so "9-02", "902" and "0902" share a token."""
    if not s:
        return ""
    out = []
    for m in _NUMRUN.findall(s):
        groups = [g.lstrip("0") or "0" for g in _DIGITS.findall(m)]
        out += groups
        if len(groups) > 1:
            out.append("".join(groups))
    return " ".join(out)


def add_norm(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("business_name").map_elements(lambda s: norm_text(s, "name"), return_dtype=pl.Utf8).alias("name_n"),
        pl.col("business_address").map_elements(lambda s: norm_text(s, "addr"), return_dtype=pl.Utf8).alias("addr_n"),
        pl.col("business_address").map_elements(num_sig, return_dtype=pl.Utf8).alias("num_n"),
    )
