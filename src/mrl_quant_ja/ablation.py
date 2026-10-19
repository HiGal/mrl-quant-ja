"""Why int8 code·code scoring loses quality: split it into centering and stretching.

sentence-transformers maps each dimension i from [lo_i, hi_i] onto [-128, 127]:
    code_i ≈ (x_i - m_i) / s_i,   m_i = (lo_i + hi_i) / 2,   s_i = (hi_i - lo_i) / 255
so the int8 dot product is (up to rounding) a dot product of *centered* and *stretched*
vectors. Scoring each distortion on its own shows which one costs the quality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from ranx import Qrels

from .compress import quantize, truncate
from .data import load_jqara
from .encode import load_embeddings
from .evaluate import score
from .search import full_search, to_run


def run_int8_ablation(cfg: dict[str, Any], model_key: str = "eg2") -> pd.DataFrame:
    data = load_jqara(cfg["dataset"]["name"], cfg["dataset"]["split"])
    qrels = Qrels(data.qrels)
    pids = data.corpus.pid.to_numpy()
    queries, corpus = load_embeddings(cfg, model_key)
    corpus = np.asarray(corpus)

    def ndcg(q: np.ndarray, d: np.ndarray) -> float:
        s, i = full_search(q.astype(np.float32), d.astype(np.float32), "float32", k=cfg["eval"]["topk"])
        return score(qrels, to_run(data.queries.qid, pids, s, i), ["ndcg@10"])["ndcg@10"]

    rows = []
    for dim in cfg["grid"]["dims"]:
        c, q = truncate(corpus, dim), truncate(queries, dim)
        lo, hi = c.min(0), c.max(0)
        m, s = (lo + hi) / 2, (hi - lo) / 255
        rows.append({
            "model": model_key,
            "dim": dim,
            "float32": ndcg(q, c),
            "centered": ndcg(q - m, c - m),
            "stretched": ndcg(q / s, c / s),
            "centered+stretched": ndcg((q - m) / s, (c - m) / s),
            "int8 code·code": ndcg(quantize(q, "int8", calibration=c), quantize(c, "int8", calibration=c)),
            # How far the corpus sits from the origin, relative to its spread.
            "mean_offset_over_std": float(np.abs(c.mean(0)).mean() / c.std(0).mean()),
        })
        print(f"[ablation] {model_key} {dim}d " + " ".join(f"{k}={v:.4f}" for k, v in rows[-1].items()
                                                          if isinstance(v, float)), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(cfg["results_dir"] / "int8_ablation.csv", index=False)
    return df
