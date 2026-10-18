"""Run the dims x precisions grid, baselines and RRF hybrid; write results/grid.csv."""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)  # ranx ships un-escaped LaTeX strings
    from ranx import Qrels, Run, evaluate, fuse

from .bm25_ja import run_bm25
from .compress import SEARCH_AS, bytes_per_vector, compress, truncate
from .data import JQaRA, load_jqara
from .encode import emb_dir, load_embeddings
from .search import full_search, rerank_candidates, rescore, to_run

RunDict = dict[str, dict[str, float]]


def score(qrels: Qrels, run: RunDict, metrics: list[str]) -> dict[str, float]:
    res = evaluate(qrels, Run(run), metrics, make_comparable=True)
    if not isinstance(res, dict):  # ranx returns a bare float for a single metric
        res = {metrics[0]: res}
    return {m: float(v) for m, v in res.items()}


def _rerank_run(data: JQaRA, q: np.ndarray, d: np.ndarray, precision: str, cand_idx) -> RunDict:
    pids = data.corpus.pid.to_numpy()
    scores = rerank_candidates(q, d, precision, cand_idx)
    return {
        qid: {str(pids[j]): float(s) for j, s in zip(c, sc)}
        for qid, c, sc in zip(data.queries.qid, cand_idx, scores)
    }


def dense_runs(
    cfg: dict[str, Any], data: JQaRA, queries: np.ndarray, corpus: np.ndarray, dim: int, precision: str,
    rescore_with: str | None, cand_idx,
) -> dict[str, RunDict]:
    """Runs for one grid cell in both evaluation modes."""
    k = cfg["eval"]["topk"]
    pids = data.corpus.pid.to_numpy()
    qc, dc = compress(queries, corpus, dim, precision)
    search_as = SEARCH_AS[precision]
    runs: dict[str, RunDict] = {}

    if rescore_with is None:
        runs["rerank"] = _rerank_run(data, qc, dc, search_as, cand_idx)
        s, i = full_search(qc, dc, search_as, k=k)
    else:
        # Binary first stage over the full corpus, then float query x {fp32, int8} docs.
        _, i = full_search(qc, dc, search_as, k=cfg["grid"]["rescore"]["k"])
        q_rich = truncate(queries, dim)
        _, d_rich = compress(queries, corpus, dim, rescore_with)
        s, i = rescore(q_rich, d_rich, i)
        s, i = s[:, :k], i[:, :k]
        # In rerank mode every candidate gets re-scored, so binary+rescore == the rich
        # precision by construction; not reported.
    runs["full"] = to_run(data.queries.qid, pids, s, i)
    return runs


def grid_cells(cfg: dict[str, Any], model_key: str):
    g = cfg["grid"]
    if not cfg["models"][model_key].get("grid", True):
        b = g["baseline_config"]
        yield b["dim"], b["precision"], None
        return
    for dim in g["dims"]:
        for p in g["precisions"]:
            yield dim, p, None
            if p == "ubinary":
                for r in g["rescore"]["with"]:
                    yield dim, p, r


def run_all(cfg: dict[str, Any]) -> pd.DataFrame:
    data = load_jqara(cfg["dataset"]["name"], cfg["dataset"]["split"])
    qrels = Qrels(data.qrels)
    metrics = cfg["eval"]["metrics"]
    modes = cfg["eval"]["modes"]
    pid_index = data.pid_index
    cand_idx = [np.array([pid_index[p] for p in data.candidates[q]]) for q in data.queries.qid]
    rows: list[dict[str, Any]] = []

    bm25 = run_bm25(cfg)
    for mode in modes:
        rows.append({"model": "bm25", "label": "BM25 (MeCab)", "dim": None, "precision": "sparse",
                     "rescore": None, "mode": mode, **score(qrels, bm25[mode], metrics)})

    dense_cache: dict[tuple, dict[str, RunDict]] = {}
    for key, mcfg in cfg["models"].items():
        if not (emb_dir(cfg, key) / "corpus.npy").exists():
            print(f"[eval] {key}: no cached embeddings, skipping (run `mrlqj encode --models {key}`)")
            continue
        queries, corpus = load_embeddings(cfg, key)
        corpus = np.asarray(corpus)
        for dim, precision, rescore_with in grid_cells(cfg, key):
            t = time.time()
            runs = dense_runs(cfg, data, queries, corpus, dim, precision, rescore_with, cand_idx)
            dense_cache[(key, dim, precision, rescore_with)] = runs
            for mode, run in runs.items():
                if mode not in modes:
                    continue
                rows.append({"model": key, "label": mcfg["label"], "dim": dim, "precision": precision,
                             "rescore": rescore_with, "mode": mode, **score(qrels, run, metrics)})
            tag = f"{precision}+{rescore_with}" if rescore_with else precision
            full = rows[-1]
            print(f"[eval] {key} {dim:>3}d {tag:<16} full nDCG@10={full['ndcg@10']:.4f} "
                  f"R@100={full['recall@100']:.4f} ({time.time() - t:.1f}s)", flush=True)

    # Hybrid BM25 + dense with reciprocal rank fusion.
    for h in cfg["hybrid"]["dense"]:
        key = (h["model"], h["dim"], h["precision"], h.get("rescore"))
        if key not in dense_cache:
            continue
        tag = f"{h['precision']}+{h['rescore']}" if h.get("rescore") else h["precision"]
        for mode in modes:
            dense = dense_cache[key].get(mode)
            if dense is None:
                continue
            fused = fuse([Run(bm25[mode]), Run(dense)], norm=None, method="rrf",
                         params={"k": cfg["hybrid"]["rrf_k"]})
            fused_dict = {q: dict(sorted(v.items(), key=lambda x: -x[1])[: cfg["eval"]["topk"]])
                          for q, v in fused.to_dict().items()}
            rows.append({"model": f"hybrid:{h['model']}", "label": f"BM25 + {cfg['models'][h['model']]['label']} (RRF)",
                         "dim": h["dim"], "precision": h["precision"], "rescore": h.get("rescore"),
                         "mode": mode, **score(qrels, fused_dict, metrics)})
            print(f"[eval] hybrid {h['dim']}d {tag} {mode} nDCG@10={rows[-1]['ndcg@10']:.4f}")

    df = pd.DataFrame(rows)
    df["bytes_per_vector"] = [
        bytes_per_vector(int(d), p) if p != "sparse" and not str(m).startswith("hybrid")
        else np.nan
        for d, p, m in zip(df["dim"].fillna(0), df["precision"], df["model"])
    ]
    df["mb_per_1m_docs"] = df["bytes_per_vector"]  # bytes/vector x 1e6 docs / 1e6 bytes-per-MB
    out = cfg["results_dir"] / "grid.csv"
    df.to_csv(out, index=False)
    print(f"[eval] wrote {out} ({len(df)} rows)")
    return df
