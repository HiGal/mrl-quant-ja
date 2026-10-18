"""Matryoshka truncation + scalar/binary quantization of cached embeddings."""

from __future__ import annotations

import numpy as np
from sentence_transformers.quantization import quantize_embeddings

PRECISIONS = ("float32", "int8", "int8_deq", "ubinary")

# How each precision is *searched*: int8_deq keeps int8 codes on disk but scores a float
# query against the dequantized docs, so search runs on the float32 path.
SEARCH_AS = {"float32": "float32", "int8": "int8", "int8_deq": "float32", "ubinary": "ubinary"}


def truncate(emb: np.ndarray, dim: int) -> np.ndarray:
    """Keep the leading `dim` Matryoshka dimensions and L2-renormalize."""
    e = np.asarray(emb[:, :dim], dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-12)


def quantize(emb: np.ndarray, precision: str, calibration: np.ndarray | None = None) -> np.ndarray:
    """float32 passthrough, int8 (per-dim min/max from `calibration`), or packed ubinary."""
    if precision == "float32":
        return emb
    if precision == "int8":
        # Queries must be quantized with the *corpus* ranges, otherwise the two sides
        # live on different grids.
        return quantize_embeddings(emb, precision="int8", calibration_embeddings=calibration)
    if precision == "ubinary":
        return quantize_embeddings(emb, precision="ubinary")
    raise ValueError(f"unknown precision {precision!r}")


def dequantize_int8(codes: np.ndarray, calibration: np.ndarray) -> np.ndarray:
    """Invert sentence-transformers' per-dimension min/max int8 mapping."""
    lo, hi = calibration.min(0), calibration.max(0)
    return (codes.astype(np.float32) + 128) * ((hi - lo) / 255) + lo


def bytes_per_vector(dim: int, precision: str) -> float:
    return {"float32": 4 * dim, "int8": dim, "int8_deq": dim, "ubinary": dim / 8}[precision]


def compress(
    queries: np.ndarray, corpus: np.ndarray, dim: int, precision: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (query codes, corpus codes) for one grid cell.

    int8     - both sides int8, scored code·code (the sentence-transformers recipe). The
               per-dim offsets add a document-dependent bias to every score.
    int8_deq - docs stored as the same int8 codes, dequantized at scoring time (per-dim
               scale/offset = 2 x dim floats for the whole index) and scored with the
               float query. Same storage, no bias.
    """
    c = truncate(corpus, dim)
    q = truncate(queries, dim)
    if precision == "int8_deq":
        return q, dequantize_int8(quantize(c, "int8", calibration=c), c)
    return quantize(q, precision, calibration=c), quantize(c, precision, calibration=c)
