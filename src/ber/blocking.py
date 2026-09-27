"""Candidate generation: per-country top-k by a joint name+address TF-IDF cosine."""
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

VIEWS = {
    "name": dict(col="name_n", analyzer="char_wb", ngram_range=(3, 3), max_df=0.005),
    "addr": dict(col="addr_n", analyzer="word", token_pattern=r"\S+", max_df=0.005),
    "addr3": dict(col="addr_n", analyzer="char_wb", ngram_range=(3, 3), max_df=0.005),
    "num": dict(col="num_n", analyzer="word", token_pattern=r"\S+", max_df=0.005),
    "nword": dict(col="name_n", analyzer="word", token_pattern=r"\S+", max_df=0.03),
}
FIT_SAMPLE = 1_000_000


def _transform(vec: TfidfVectorizer, texts: list[str], n_jobs: int = 16) -> sparse.csr_matrix:
    step = -(-len(texts) // n_jobs)
    parts = Parallel(n_jobs=n_jobs)(delayed(vec.transform)(texts[i:i + step]) for i in range(0, len(texts), step))
    return sparse.vstack(parts).tocsr()


def fit_view(texts: pl.Series, view: str) -> TfidfVectorizer:
    cfg = {k: v for k, v in VIEWS[view].items() if k != "col"}
    sample = texts.sample(min(FIT_SAMPLE, len(texts)), seed=0).to_list()
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True, dtype=np.float32, **cfg).fit(sample)
    vec.stop_words_ = None  # every pruned term; only for introspection, but copied to each worker
    return vec


def embed(queries: pl.DataFrame, pool: pl.DataFrame, views: list[str]):
    """Per-view L2-normalised TF-IDF matrices for queries and pool (fit on both, no labels)."""
    out = {}
    for v in views:
        col = VIEWS[v]["col"]
        vec = fit_view(pl.concat([queries[col], pool[col]]), v)
        out[v] = (_transform(vec, queries[col].to_list()), _transform(vec, pool[col].to_list()))
    return out


def joint(mats: dict, weights: dict[str, float]):
    q = sparse.hstack([mats[v][0] * np.sqrt(w) for v, w in weights.items()]).tocsr()
    p = sparse.hstack([mats[v][1] * np.sqrt(w) for v, w in weights.items()]).tocsr()
    return q, p


def topn(q: sparse.csr_matrix, p: sparse.csr_matrix, k: int, chunk: int = 20000):
    p_t = p.T.tocsr()
    rows, cols, vals = [], [], []
    for i in range(0, q.shape[0], chunk):
        m = sp_matmul_topn(q[i:i + chunk], p_t, top_n=k, threshold=1e-4, n_threads=16).tocoo()
        rows.append(m.row + i), cols.append(m.col), vals.append(m.data)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals).astype(np.float32)


def ranks(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    order = np.lexsort((-v, r))
    rank = np.empty(len(r), dtype=np.int16)
    starts = np.r_[0, np.flatnonzero(np.diff(r[order])) + 1]
    rank[order] = np.arange(len(r)) - np.repeat(starts, np.diff(np.r_[starts, len(r)]))
    return rank


def rowwise_cos(a: sparse.csr_matrix, b: sparse.csr_matrix, ia: np.ndarray, ib: np.ndarray, chunk: int = 2_000_000):
    """cos(a[ia[j]], b[ib[j]]) for each j; rows are already L2-normalised."""
    out = np.empty(len(ia), dtype=np.float32)
    for i in range(0, len(ia), chunk):
        out[i:i + chunk] = np.asarray(a[ia[i:i + chunk]].multiply(b[ib[i:i + chunk]]).sum(axis=1)).ravel()
    return out
