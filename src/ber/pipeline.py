"""End-to-end steps: blocking -> cheap-feature pruning -> pair model -> selection."""
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from joblib import Parallel, delayed

from ber import blocking as B
from ber.features import group_features, sibling_features, string_features

K_JOINT = 30
K_EMPTY = 10
K_NUM = 10
WEIGHTS = {"name": 0.3, "addr": 0.7}
WEIGHTS_NUM = {"name": 0.3, "addr": 0.5, "num": 0.2}

A_FEATS = [
    "joint", "name_cos", "addr_cos", "num_cos", "j_rank", "e_rank", "n_rank", "joint_gap_s1", "joint_rk_s1", "joint_gap_cand",
    "joint_rk_cand", "name_cos_gap_s1", "name_cos_rk_s1", "name_cos_gap_cand", "name_cos_rk_cand",
    "addr_cos_gap_s1", "addr_cos_gap_cand", "num_cos_gap_s1", "num_cos_gap_cand", "n_cand_s1", "n_s1_cand", "is_s3", "c_addr_empty",
    "q_addr_len", "c_addr_len", "q_name_len", "c_name_len", "c_indic",
]
B_EXTRA = ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_jw", "a_tset", "a_tsort", "a_partial",
           "num_jac", "num_first", "p_a", "p_a_gap_s1", "p_a_rk_s1", "p_a_gap_cand", "p_a_rk_cand", "n_cand_b",
           "sib_name", "sib_addr", "sib_both", "sib_n_strong"]
B_FEATS = A_FEATS + B_EXTRA


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def block(queries: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Per country, a union of three searches: joint name+address top-K_JOINT; name-only
    top-K_EMPTY among pool records without an address (the joint score cannot rank those);
    and name+address+house-number top-K_NUM."""
    out = []
    for (country,), qc in queries.group_by("country", maintain_order=True):
        pc = pool.filter(pl.col("country") == country)
        if pc.height == 0:
            continue
        mats = B.embed(qc, pc, list(WEIGHTS_NUM))
        r, c, v = B.topn(*B.joint(mats, WEIGHTS), K_JOINT)
        joint = pl.DataFrame({"qi": r, "pi": c, "j_rank": B.ranks(r, v)})
        r, c2, v = B.topn(*B.joint(mats, WEIGHTS_NUM), K_NUM)
        joint = joint.join(pl.DataFrame({"qi": r, "pi": c2.astype(c.dtype), "n_rank": B.ranks(r, v)}),
                           on=["qi", "pi"], how="full", coalesce=True)

        empty_idx = np.flatnonzero((pc["addr_n"] == "").to_numpy())
        if len(empty_idx):
            re_, ce, ve = B.topn(mats["name"][0], mats["name"][1][empty_idx], K_EMPTY)
            empty = pl.DataFrame({"qi": re_, "pi": empty_idx[ce].astype(c.dtype), "e_rank": B.ranks(re_, ve)})
            pairs = joint.join(empty, on=["qi", "pi"], how="full", coalesce=True)
        else:
            pairs = joint.with_columns(pl.lit(None, pl.Int16).alias("e_rank"))
        qi, pi = pairs["qi"].to_numpy(), pairs["pi"].to_numpy()
        name_cos = B.rowwise_cos(*mats["name"], qi, pi)
        addr_cos = B.rowwise_cos(*mats["addr"], qi, pi)
        num_cos = B.rowwise_cos(*mats["num"], qi, pi)
        del mats
        out.append(pl.DataFrame({
            "s1_id": qc["entity_id"].gather(qi),
            "cand_id": pc["entity_id"].gather(pi),
            "joint": WEIGHTS["name"] * name_cos + WEIGHTS["addr"] * addr_cos,
            "name_cos": name_cos,
            "addr_cos": addr_cos,
            "num_cos": num_cos,
            "j_rank": pairs["j_rank"].fill_null(K_JOINT),
            "e_rank": pairs["e_rank"].fill_null(K_EMPTY),
            "n_rank": pairs["n_rank"].fill_null(K_NUM),
        }))
        log(f"blocked {country}: {qc.height} queries, {pc.height} pool, {pairs.height} pairs")
    return pl.concat(out)


def record_stats(recs: pl.DataFrame) -> pl.DataFrame:
    return recs.select(
        "entity_id",
        pl.col("name_n").str.len_chars().alias("name_len"),
        pl.col("addr_n").str.len_chars().alias("addr_len"),
        pl.col("business_name").str.contains(r"[ऀ-ൿ]").alias("indic"),
    )


def cheap_features(pairs: pl.DataFrame, stats: pl.DataFrame) -> pl.DataFrame:
    df = (
        pairs.join(stats.select(pl.col("entity_id").alias("s1_id"), pl.col("name_len").alias("q_name_len"),
                                pl.col("addr_len").alias("q_addr_len")), on="s1_id")
        .join(stats.select(pl.col("entity_id").alias("cand_id"), pl.col("name_len").alias("c_name_len"),
                           pl.col("addr_len").alias("c_addr_len"), pl.col("indic").alias("c_indic")), on="cand_id")
        .with_columns(
            pl.col("cand_id").str.starts_with("S3-").alias("is_s3"),
            (pl.col("c_addr_len") == 0).alias("c_addr_empty"),
        )
    )
    return group_features(df, ["joint", "name_cos", "addr_cos", "num_cos"])


def full_features(pairs: pl.DataFrame, recs: pl.DataFrame, n_jobs: int = 16) -> pl.DataFrame:
    side = recs.select("entity_id", "name_n", "addr_n")
    txt = (
        pairs.select("s1_id", "cand_id")
        .join(side.rename({"entity_id": "s1_id", "name_n": "name_n_q", "addr_n": "addr_n_q"}), on="s1_id", how="left")
        .join(side.rename({"entity_id": "cand_id", "name_n": "name_n_c", "addr_n": "addr_n_c"}), on="cand_id", how="left")
        .drop("s1_id", "cand_id")
    )
    step = -(-txt.height // n_jobs)
    parts = Parallel(n_jobs=n_jobs)(
        delayed(string_features)(txt.slice(i, step)) for i in range(0, txt.height, step)
    )
    df = pl.concat([pairs, pl.concat(parts)], how="horizontal")
    df = group_features(df, ["p_a"]).with_columns(pl.len().over("s1_id").alias("n_cand_b"))
    return sibling_features(df, recs, n_jobs)


def prune(df: pl.DataFrame, model: lgb.Booster, tau: float, max_cands: int) -> pl.DataFrame:
    df = df.with_columns(pl.Series("p_a", predict(model, df, A_FEATS)))
    df = df.filter(pl.col("p_a") >= tau)
    return df.filter(pl.col("p_a").rank("ordinal", descending=True).over("s1_id") <= max_cands)


def to_numpy(df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return df.select([pl.col(f).cast(pl.Float32) for f in feats]).to_numpy()


def train_lgb(train: pl.DataFrame, valid: pl.DataFrame, feats: list[str], rounds: int = 5000) -> lgb.Booster:
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=16, seed=0)
    dtr = lgb.Dataset(to_numpy(train, feats), train["y"].to_numpy(), feature_name=feats)
    dva = lgb.Dataset(to_numpy(valid, feats), valid["y"].to_numpy(), reference=dtr)
    return lgb.train(params, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])


def predict(model: lgb.Booster, df: pl.DataFrame, feats: list[str]) -> np.ndarray:
    return model.predict(to_numpy(df, feats), num_threads=16).astype(np.float32)
