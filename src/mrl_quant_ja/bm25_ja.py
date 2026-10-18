"""BM25 baseline with MeCab (fugashi + unidic-lite) tokenization."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import bm25s
import numpy as np

from .data import load_jqara

# UniDic POS1 tags that carry no lexical signal.
_DROP_POS = {"補助記号", "空白"}


class JapaneseTokenizer:
    def __init__(self) -> None:
        import fugashi

        self.tagger = fugashi.Tagger()

    def __call__(self, text: str) -> list[str]:
        return [w.surface.lower() for w in self.tagger(text) if w.feature.pos1 not in _DROP_POS]


def runs_dir(cfg: dict[str, Any]) -> Path:
    d = cfg["cache_dir"] / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_bm25(cfg: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    out = runs_dir(cfg)
    paths = {m: out / f"bm25_{m}.json" for m in ("rerank", "full")}
    if all(p.exists() for p in paths.values()):
        print("[bm25] cached")
        return {m: json.loads(p.read_text()) for m, p in paths.items()}

    data = load_jqara(cfg["dataset"]["name"], cfg["dataset"]["split"])
    tok = JapaneseTokenizer()
    t0 = time.time()
    docs = [tok(f"{t} {x}") for t, x in zip(data.corpus.title, data.corpus.text)]
    queries = [tok(q) for q in data.queries.question]
    print(f"[bm25] tokenized {len(docs)} docs in {time.time() - t0:.0f}s")

    # bm25s wants integer ids; build the vocabulary ourselves so MeCab tokens are kept as-is.
    vocab: dict[str, int] = {}
    doc_ids = [[vocab.setdefault(w, len(vocab)) for w in d] for d in docs]
    retriever = bm25s.BM25(k1=cfg["bm25"]["k1"], b=cfg["bm25"]["b"])
    retriever.index(bm25s.tokenization.Tokenized(ids=doc_ids, vocab=vocab), show_progress=False)

    pids = data.corpus.pid.to_numpy()
    pid_index = data.pid_index
    k = cfg["eval"]["topk"]
    runs: dict[str, dict[str, dict[str, float]]] = {"rerank": {}, "full": {}}
    for qid, q in zip(data.queries.qid, queries):
        q_ids = [vocab[w] for w in q if w in vocab]
        scores = retriever.get_scores_from_ids(q_ids) if q_ids else np.zeros(len(pids), dtype=np.float32)
        cand = [pid_index[p] for p in data.candidates[qid]]
        runs["rerank"][qid] = {str(pids[j]): float(scores[j]) for j in cand}
        top = np.argpartition(-scores, k)[:k]
        runs["full"][qid] = {str(pids[j]): float(scores[j]) for j in top}

    for m, p in paths.items():
        p.write_text(json.dumps(runs[m]))
    print(f"[bm25] done in {time.time() - t0:.0f}s")
    return runs
