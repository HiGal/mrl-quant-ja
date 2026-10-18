"""CPU query-encode latency: PyTorch fp32 vs ONNX Runtime fp32 vs ONNX Runtime int8.

Also measures what int8 weight quantization of the *query encoder* does to quality:
all JQaRA queries are re-encoded with each backend and scored against the cached
PyTorch corpus embeddings.

    uv run mrlqj latency
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from mrl_quant_ja.data import load_jqara
from mrl_quant_ja.encode import emb_dir, format_queries, load_model


class TextEncoder(nn.Module):
    """input_ids, attention_mask -> L2-normalized mean-pooled 768d embedding."""

    def __init__(self, lm: nn.Module) -> None:
        super().__init__()
        self.lm = lm

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.lm(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _resave_single_file(path: Path) -> None:
    """Fold per-tensor external data files into one `<name>.data` next to the model."""
    import onnx

    m = onnx.load(str(path))
    for f in path.parent.iterdir():
        if f != path:
            f.unlink()
    onnx.save_model(m, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=path.name + ".data", size_threshold=1024)


def export_onnx(model, out_dir: Path) -> tuple[Path, Path]:
    """Export the text tower (+ mean pooling + L2 norm) and a dynamic-int8 copy.

    int8 = per-channel int8 weights for every MatMul and for the 262k x 512 token
    embedding table (Gather), activations quantized on the fly.
    """
    import shutil

    from onnxruntime.quantization import QuantType, quantize_dynamic

    shutil.rmtree(out_dir, ignore_errors=True)
    fp32, int8 = out_dir / "fp32" / "model.onnx", out_dir / "int8" / "model.onnx"
    fp32.parent.mkdir(parents=True)
    int8.parent.mkdir(parents=True)

    enc = TextEncoder(model[0].model.language_model).eval()
    sample = model.tokenizer(["task: search result | query: テスト", "短い"], padding=True, return_tensors="pt")
    args = (sample["input_ids"], sample["attention_mask"])
    batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq")
    with torch.no_grad():
        torch.onnx.export(
            enc, args, str(fp32), dynamo=True, external_data=True,
            input_names=["input_ids", "attention_mask"], output_names=["pooled_output"],
            dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
        )
    _resave_single_file(fp32)
    quantize_dynamic(
        str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm", "Gather"], use_external_data_format=True,
    )
    _resave_single_file(int8)
    return fp32, int8


class TorchBackend:
    name = "PyTorch fp32"

    def __init__(self, model) -> None:
        self.model = model

    def __call__(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=len(texts), normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)


class OrtBackend:
    def __init__(self, name: str, path: Path, tokenizer, threads: int) -> None:
        import onnxruntime as ort

        self.name, self.tokenizer = name, tokenizer
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def __call__(self, texts: list[str]) -> np.ndarray:
        t = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="np")
        return self.sess.run(None, {"input_ids": t["input_ids"].astype(np.int64),
                                    "attention_mask": t["attention_mask"].astype(np.int64)})[0]


def time_backend(fn, texts: list[str], batch_size: int, warmup: int) -> dict[str, float]:
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    batches = [b for b in batches if len(b) == batch_size]
    for b in batches[:warmup]:
        fn(b)
    lat = []
    for b in batches:
        t = time.perf_counter()
        fn(b)
        lat.append((time.perf_counter() - t) * 1000)
    q = sorted(lat)
    return {
        "batch_ms_p50": statistics.median(lat),
        "batch_ms_p95": q[int(0.95 * (len(q) - 1))],
        "ms_per_query": statistics.median(lat) / batch_size,
        "queries_per_s": 1000 * batch_size / statistics.median(lat),
        "n_batches": len(lat),
    }


def encode_all(fn, texts: list[str], bs: int = 32) -> np.ndarray:
    return np.concatenate([fn(texts[i : i + bs]) for i in range(0, len(texts), bs)])


def run_latency(cfg: dict[str, Any], skip_export: bool = False) -> pd.DataFrame:
    lcfg = cfg["latency"]
    key = lcfg["model"]
    mcfg = cfg["models"][key]
    torch.set_num_threads(lcfg["threads"])
    os.environ.setdefault("OMP_NUM_THREADS", str(lcfg["threads"]))

    data = load_jqara(cfg["dataset"]["name"], cfg["dataset"]["split"])
    texts = format_queries(mcfg, data.queries)
    model = load_model(mcfg, cfg["encode"], device="cpu")

    onnx_dir = cfg["cache_dir"].parent / "onnx" / key
    fp32, int8 = onnx_dir / "fp32" / "model.onnx", onnx_dir / "int8" / "model.onnx"
    if not (skip_export and fp32.exists() and int8.exists()):
        fp32, int8 = export_onnx(model, onnx_dir)

    def size_mb(p: Path) -> float:
        return sum(f.stat().st_size for f in p.parent.iterdir()) / 1e6

    backends = [
        (TorchBackend(model), size_mb_torch(model)),
        (OrtBackend("ONNX Runtime fp32", fp32, model.tokenizer, lcfg["threads"]), size_mb(fp32)),
        (OrtBackend("ONNX Runtime int8", int8, model.tokenizer, lcfg["threads"]), size_mb(int8)),
    ]

    rng = np.random.default_rng(0)
    sample = [texts[i] for i in rng.choice(len(texts), size=lcfg["n_queries"], replace=False)]
    rows = []
    for fn, mb in backends:
        for bs in lcfg["batch_sizes"]:
            r = time_backend(fn, sample, bs, lcfg["warmup"])
            rows.append({"backend": fn.name, "batch_size": bs, "model_mb": round(mb), **r})
            print(f"[latency] {fn.name:<18} bs={bs:<3} {r['batch_ms_p50']:.1f} ms/batch "
                  f"({r['ms_per_query']:.2f} ms/query)", flush=True)

    # Query-side fidelity: re-encode every query with each backend.
    ref = None
    fidelity = []
    out = emb_dir(cfg, key)
    for fn, _ in backends:
        q = encode_all(fn, texts)
        if ref is None:
            ref = q
        tag = fn.name.split()[-1] if fn.name.startswith("ONNX") else "torch"
        np.save(out / f"queries_cpu_{'onnx_' if fn.name.startswith('ONNX') else ''}{tag}.npy", q)
        cos = (q * ref).sum(1)
        fidelity.append({"backend": fn.name, "cos_vs_torch_mean": float(cos.mean()),
                         "cos_vs_torch_min": float(cos.min())})

    # Does an int8 query encoder change retrieval quality? Score each query set against
    # the cached PyTorch corpus (768d fp32 and the 256d binary+int8-rescore config).
    if (out / "corpus.npy").exists():
        from ranx import Qrels

        from mrl_quant_ja.evaluate import dense_runs, score

        corpus = np.load(out / "corpus.npy")
        qrels = Qrels(data.qrels)
        pid_index = data.pid_index
        cand_idx = [np.array([pid_index[p] for p in data.candidates[q]]) for q in data.queries.qid]
        for f in fidelity:
            tag = f["backend"].split()[-1] if f["backend"].startswith("ONNX") else "torch"
            q = np.load(out / f"queries_cpu_{'onnx_' if f['backend'].startswith('ONNX') else ''}{tag}.npy")
            for dim, prec, resc in [(768, "float32", None), (256, "int8_deq", None), (512, "ubinary", "int8_deq")]:
                runs = dense_runs(cfg, data, q, corpus, dim, prec, resc, cand_idx)
                name = f"{dim}d {prec}" + (f"+{resc}" if resc else "")
                f[f"ndcg@10 full {name}"] = score(qrels, runs["full"], ["ndcg@10"])["ndcg@10"]

    lat = pd.DataFrame(rows)
    lat.to_csv(cfg["results_dir"] / "latency.csv", index=False)
    pd.DataFrame(fidelity).to_csv(cfg["results_dir"] / "onnx_fidelity.csv", index=False)
    (cfg["results_dir"] / "hardware.json").write_text(json.dumps(hardware_info(lcfg["threads"]), indent=2))
    print(lat.to_string(index=False))
    return lat


def size_mb_torch(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def hardware_info(threads: int) -> dict[str, Any]:
    import platform
    import subprocess

    import onnxruntime

    def sysctl(k: str) -> str:
        try:
            return subprocess.check_output(["sysctl", "-n", k], text=True).strip()
        except Exception:
            return ""

    return {
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cores": os.cpu_count(),
        "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2**30, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "threads": threads,
    }
