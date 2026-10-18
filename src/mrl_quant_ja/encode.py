"""Encode JQaRA once at full 768d float32 and cache to disk.

Every downstream config (truncation, quantization) is derived from this cache, so the
expensive model pass happens exactly once per model. Corpus encoding is chunked and
resumable: an interrupted run picks up from the last finished chunk.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .config import emb_dir
from .data import JQaRA

CHUNK = 4096


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(mcfg: dict[str, Any], ecfg: dict[str, Any], device: str | None = None) -> SentenceTransformer:
    model = SentenceTransformer(
        mcfg["id"],
        device=device or pick_device(),
        model_kwargs={"torch_dtype": getattr(torch, ecfg.get("dtype", "float32"))},
        config_kwargs=mcfg.get("config_kwargs") or {},
    )
    model.max_seq_length = ecfg.get("max_seq_length", 512)
    return model


def format_docs(mcfg: dict[str, Any], corpus) -> list[str]:
    tpl = mcfg["doc_template"]
    return [tpl.format(title=t, text=x) for t, x in zip(corpus.title, corpus.text)]


def format_queries(mcfg: dict[str, Any], queries) -> list[str]:
    return [mcfg["query_prompt"] + q for q in queries.question]


def _encode(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return emb.astype(np.float32)


def encode_model(cfg: dict[str, Any], model_key: str, data: JQaRA, device: str | None = None) -> Path:
    mcfg, ecfg = cfg["models"][model_key], cfg["encode"]
    out = emb_dir(cfg, model_key)
    if (out / "corpus.npy").exists() and (out / "queries.npy").exists():
        print(f"[encode] {model_key}: cached at {out}")
        return out

    model = load_model(mcfg, ecfg, device)
    print(f"[encode] {model_key}: {mcfg['id']} on {model.device}")

    t0 = time.time()
    q = _encode(model, format_queries(mcfg, data.queries), ecfg["batch_size"])
    np.save(out / "queries.npy", q)

    docs = format_docs(mcfg, data.corpus)
    chunk_dir = out / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    n_chunks = (len(docs) + CHUNK - 1) // CHUNK
    for i in range(n_chunks):
        path = chunk_dir / f"{i:04d}.npy"
        if path.exists():
            continue
        t = time.time()
        np.save(path, _encode(model, docs[i * CHUNK : (i + 1) * CHUNK], ecfg["batch_size"]))
        rate = min(CHUNK, len(docs) - i * CHUNK) / (time.time() - t)
        print(f"[encode] {model_key}: chunk {i + 1}/{n_chunks} ({rate:.0f} docs/s)", flush=True)

    corpus = np.concatenate([np.load(chunk_dir / f"{i:04d}.npy") for i in range(n_chunks)])
    assert corpus.shape[0] == len(docs)
    np.save(out / "corpus.npy", corpus)
    for p in chunk_dir.glob("*.npy"):
        p.unlink()
    chunk_dir.rmdir()

    meta = {
        "model_id": mcfg["id"],
        "device": str(model.device),
        "dtype": ecfg.get("dtype", "float32"),
        "n_docs": len(docs),
        "n_queries": len(q),
        "dim": int(corpus.shape[1]),
        "seconds": round(time.time() - t0, 1),
        "pids": "corpus order == data.corpus.pid (sorted)",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[encode] {model_key}: done in {meta['seconds']}s")
    return out


def load_embeddings(cfg: dict[str, Any], model_key: str) -> tuple[np.ndarray, np.ndarray]:
    d = emb_dir(cfg, model_key)
    return np.load(d / "queries.npy"), np.load(d / "corpus.npy", mmap_mode="r")
