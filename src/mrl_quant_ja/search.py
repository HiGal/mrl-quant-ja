"""Exact search over compressed vectors: inner product, int8 dot, Hamming, + rescoring.

Two evaluation modes:
  * rerank - score only the 100 candidates JQaRA ships per query (official protocol).
  * full   - retrieve top-k from the pooled corpus of every test passage (~144k docs).
"""

from __future__ import annotations

import faiss
import numpy as np

# faiss and torch each bundle their own libomp on macOS; running faiss multi-threaded in
# a process that has imported torch segfaults. Exact search at this scale is fast anyway.
faiss.omp_set_num_threads(1)

Run = dict[str, dict[str, float]]


def _scores_dense(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    """q: (nq, dim), d: (nd, dim) float32 or int8 -> (nq, nd) float32 dot products."""
    return q.astype(np.float32) @ d.astype(np.float32).T


def _scores_hamming(q: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Packed uint8 bits -> negative Hamming distance (higher is better)."""
    x = np.bitwise_xor(q[:, None, :], d[None, :, :])
    return -np.bitwise_count(x).sum(-1).astype(np.float32)


def pairwise_scores(q: np.ndarray, d: np.ndarray, precision: str) -> np.ndarray:
    return _scores_hamming(q, d) if precision == "ubinary" else _scores_dense(q, d)


def _topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    s = np.take_along_axis(scores, idx, 1)
    order = np.argsort(-s, axis=1, kind="stable")
    return np.take_along_axis(s, order, 1), np.take_along_axis(idx, order, 1)


def full_search(
    q: np.ndarray, d: np.ndarray, precision: str, k: int = 100, chunk: int = 16384
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k over the whole corpus. Returns (scores, doc indices), both (nq, k)."""
    if precision == "float32":
        index = faiss.IndexFlatIP(d.shape[1])
        index.add(np.ascontiguousarray(d, dtype=np.float32))
        return index.search(np.ascontiguousarray(q, dtype=np.float32), k)
    if precision == "ubinary":
        index = faiss.IndexBinaryFlat(d.shape[1] * 8)
        index.add(np.ascontiguousarray(d))
        dist, idx = index.search(np.ascontiguousarray(q), k)
        return -dist.astype(np.float32), idx
    # int8: exact dot products, streamed over corpus chunks to bound memory.
    best_s = np.full((len(q), 0), -np.inf, dtype=np.float32)
    best_i = np.zeros((len(q), 0), dtype=np.int64)
    qf = q.astype(np.float32)
    for start in range(0, len(d), chunk):
        s = qf @ np.asarray(d[start : start + chunk], dtype=np.float32).T
        s, i = _topk(s, k)
        best_s = np.concatenate([best_s, s], 1)
        best_i = np.concatenate([best_i, i + start], 1)
        best_s, j = _topk(best_s, k)
        best_i = np.take_along_axis(best_i, j, 1)
    return best_s, best_i


def rescore(
    q_rich: np.ndarray, d_rich: np.ndarray, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Re-rank first-stage candidates `idx` (nq, k) with float query x richer docs."""
    d_rich = np.asarray(d_rich)
    s = np.empty(idx.shape, dtype=np.float32)
    for b in range(0, len(idx), 128):  # batches keep the (nq, k, dim) gather small
        cand = d_rich[idx[b : b + 128]].astype(np.float32)
        s[b : b + 128] = np.einsum("qd,qkd->qk", q_rich[b : b + 128].astype(np.float32), cand)
    order = np.argsort(-s, axis=1, kind="stable")
    return np.take_along_axis(s, order, 1), np.take_along_axis(idx, order, 1)


def rerank_candidates(
    q: np.ndarray, d: np.ndarray, precision: str, cand_idx: list[np.ndarray]
) -> list[np.ndarray]:
    """Per-query scores over that query's own candidate list (JQaRA protocol)."""
    return [pairwise_scores(q[i : i + 1], np.asarray(d)[c], precision)[0] for i, c in enumerate(cand_idx)]


def to_run(qids, pids: np.ndarray, scores, idx) -> Run:
    """Build a ranx-style run {qid: {pid: score}} from (scores, indices) rows."""
    run: Run = {}
    for qid, s, i in zip(qids, scores, idx):
        run[qid] = {str(pids[j]): float(v) for j, v in zip(i, s)}
    return run
