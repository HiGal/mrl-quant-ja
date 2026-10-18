"""JQaRA loading: corpus, queries, qrels and per-query candidate lists."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from datasets import load_dataset


@dataclass(frozen=True)
class JQaRA:
    corpus: pd.DataFrame            # pid, title, text  (unique passages, stable order)
    queries: pd.DataFrame           # qid, question     (stable order)
    qrels: dict[str, dict[str, int]]          # qid -> {pid: 1} (positives only)
    candidates: dict[str, list[str]]          # qid -> the 100 pids JQaRA ships for it

    @property
    def pid_index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.corpus.pid)}


@lru_cache(maxsize=2)
def load_jqara(name: str = "hotchpotch/JQaRA", split: str = "test") -> JQaRA:
    df = load_dataset(name, split=split).to_pandas()
    df = df.rename(columns={"q_id": "qid", "passage_row_id": "pid"})

    corpus = (
        df.drop_duplicates("pid")[["pid", "title", "text"]]
        .sort_values("pid", kind="stable")
        .reset_index(drop=True)
    )
    queries = df.drop_duplicates("qid")[["qid", "question"]].reset_index(drop=True)

    qrels: dict[str, dict[str, int]] = {}
    for qid, pid in df.loc[df.label > 0, ["qid", "pid"]].itertuples(index=False):
        qrels.setdefault(qid, {})[pid] = 1
    candidates = df.groupby("qid", sort=False).pid.apply(list).to_dict()
    return JQaRA(corpus=corpus, queries=queries, qrels=qrels, candidates=candidates)
