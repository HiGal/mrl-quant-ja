# EmbeddingGemma 2 on Japanese retrieval 

**How small can a Japanese search vector get?** A storage × quality × latency study of
[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)'s Matryoshka truncation
combined with int8 / binary quantization, on the Japanese RAG retrieval benchmark
[JQaRA](https://huggingface.co/datasets/hotchpotch/JQaRA).

The model card reports MRL only as MTEB averages, at full precision. This repo measures what
happens on **Japanese** retrieval when truncation is *stacked* with quantization, and what the
query encoder costs on a CPU.

![Pareto: index size vs nDCG@10](figures/pareto.png)

## TL;DR

| | config | RAM / vector | nDCG@10 (full corpus) | vs 768d fp32 |
|---|---|---:|---:|---:|
| reference | 768d float32 | 3,072 B | 0.5668 | 100% |
| lossless | 768d int8, dequantized scoring | 768 B | 0.5671 | 100% |
| **knee** | **512d binary → int8 rescore of top-100** | **64 B** | **0.5555** | **98%** |
| aggressive | 256d binary → int8 rescore | 32 B | 0.5127 | 90% |
| where it breaks | 128d binary (no rescore) | 16 B | 0.1801 | 32% |
| sparse baseline | BM25 (MeCab) | — | 0.4736 | 84% |

* **48× less RAM for a 2% loss.** 512 Matryoshka dims, sign-binarized and searched by Hamming
  distance, then the top-100 re-scored with float query × int8 docs, keeps 98% of full-precision nDCG@10.
* **int8 can be lossless, but only if you score it right.** The common recipe (int8 query codes · int8
  doc codes, as in `sentence_transformers.quantization`) loses **3.5–7.6 nDCG points**. Scoring the
  float query against *dequantized* int8 docs (same 1 byte/dim storage + 2×dim floats for
  the whole index) matches float32 at every dimension. Per-dimension min/max offsets add a
  document-dependent bias to every code·code dot product. That bias is what costs the points.
* **Matryoshka is nearly free down to 256d** (−0.7 pt at 512d, −2.8 pt at 256d) and
  **breaks at 128d** (−10 pt, below BM25 in the official protocol). The model card's
  "close to lossless down to 256 dimensions" holds for Japanese too.
* **Binary without rescoring is not production-ready below 768d**: 85% at 768d, 77% at
  512d, 60% at 256d, 32% at 128d.
* **EmbeddingGemma 1 is still better at Japanese.** It beats EmbeddingGemma 2 at every
  dimension and precision, by +0.3 to +7.3 nDCG points: ~2.5 at 768/512d float32, and the
  gap is widest for binary codes (+5 to +7 without rescoring) ([chart](figures/eg2_vs_eg1.png)). EG2's gains are in
  code and multimodal retrieval, not Japanese text.
* **ONNX Runtime int8 halves single-query CPU latency** (77 → 38 ms) and shrinks the encoder 4×
  (1.08 GB → 278 MB). It costs ~1.2 nDCG points (query cosine to fp32: 0.982 mean).

## What to use in production

1. **Default: 256d–512d int8 with asymmetric (dequantized) scoring.** That is 6–12× smaller than
   768d fp32 at 95–99% quality, and plain exact or IVF search still works. Never ship the int8 code·code dot product.
2. **RAM-bound or 10M+ docs: 512d binary in RAM + int8 on disk, rescore the top-k.**
   64 B/vector in RAM (64 MB per 1M docs) at 98% nDCG@10. With k = 100, Recall@100 drops to 0.75
   (vs 0.89), so for RAG pipelines that want recall, retrieve 4–10× more candidates before rescoring.
3. **Japanese-only text search: EmbeddingGemma 1 is the better model.** Pick EG2 when you also
   need code, images or audio in the same space. Avoid 128d for Japanese. Use the ONNX int8
   query encoder only when the ~1 pt quality cost is worth 2× lower latency.

## Results

Full tables are in [`results/RESULTS.md`](results/RESULTS.md). Raw numbers: [`results/grid.csv`](results/grid.csv),
[`results/latency.csv`](results/latency.csv), [`results/onnx_fidelity.csv`](results/onnx_fidelity.csv).

### Two evaluation modes

* **full** (primary): exact top-100 retrieval over the pooled corpus of **all 144,372 unique
  test passages** for each of the 1,667 questions. This is what a real index does, and it is
  the only mode where Recall@100 and binary-first-stage search mean anything.
* **rerank**: the official JQaRA protocol. Each question ranks only its own 100 shipped
  candidates. Comparable to the [JQaRA leaderboard](https://huggingface.co/datasets/hotchpotch/JQaRA)
  (e.g. multilingual-e5-large 0.554, bge-m3 dense 0.539, ruri-large 0.629, BM25 0.458).
  Rescoring is not reported here: re-scoring all 100 candidates equals the rescoring precision by construction.

### EmbeddingGemma 2 grid (full corpus)

| dims | float32 | int8 code·code | int8 dequantized | binary | binary → int8 rescore |
|---:|---:|---:|---:|---:|---:|
| bytes/vec | 4·d | d | d | d/8 | d/8 (+ d on disk) |
| 768 | 0.5668 | 0.5321 | **0.5671** | 0.4805 | 0.5651 |
| 512 | 0.5601 | 0.5225 | **0.5601** | 0.4387 | 0.5555 |
| 256 | 0.5393 | 0.4942 | **0.5396** | 0.3426 | 0.5127 |
| 128 | 0.4658 | 0.3901 | **0.4666** | 0.1801 | 0.3733 |

### Baselines (768d float32 unless noted)

| system | nDCG@10 full | R@100 full | nDCG@10 rerank |
|---|---:|---:|---:|
| BM25 (MeCab/unidic-lite, bm25s) | 0.4736 | 0.7410 | 0.5107 |
| EmbeddingGemma 1 | **0.5924** | **0.9196** | **0.6028** |
| EmbeddingGemma 2 | 0.5668 | 0.8959 | 0.5832 |
| BM25 + EmbeddingGemma 2, RRF (k=60) | 0.5674 | 0.8904 | 0.5875 |

Hybrid RRF barely moves nDCG here: BM25 is ~10 points behind the dense model, so fusion adds little.

### CPU query-encode latency (EmbeddingGemma 2 text tower)

Apple M1 Pro (8 threads), end-to-end including tokenization, real JQaRA queries, median of 256 queries.

| backend | model size | batch 1: ms/query | batch 32: ms/query |
|---|---:|---:|---:|
| PyTorch fp32 | 1,084 MB | 77.4 | 23.7 |
| ONNX Runtime fp32 | 1,090 MB | 63.1 | 56.2 |
| ONNX Runtime int8 (dynamic, per-channel; MatMul + embedding table) | 278 MB | **38.2** | 27.9 |

ONNX Runtime wins at batch 1 (the online-query case). At batch 32 the exported graph's padded
attention is slower than PyTorch's SDPA kernel. Use PyTorch for offline batch encoding.

## Method

* **Model**: `google/embeddinggemma-2`, text encoder only (`vision_config=None, audio_config=None`,
  271M params), float32. Queries use `task: search result | query: {q}`. Documents use
  `title: {title} | text: {text}`, per the model card. Baseline EmbeddingGemma 1 =
  `unsloth/embeddinggemma-300m`, an ungated mirror of the gated `google/embeddinggemma-300m`.
* **Encode once**: every passage and query is encoded at 768d fp32 and cached
  (`cache/embeddings/<model>/`). All 24 grid cells per model are derived from that cache.
* **Compress**: truncate to the leading *d* dims → L2-renormalize → quantize.
  int8 = per-dimension min/max calibrated on the corpus (queries use the corpus ranges).
  binary = sign bit, packed (`ubinary`).
* **Search**: exact. FAISS `IndexFlatIP` (float), `IndexBinaryFlat` (Hamming), exact int8 dot.
  Rescoring re-ranks the binary top-100 with float query × {fp32, dequantized int8} docs.
* **Metrics**: ranx nDCG@10, Recall@100, MRR@10. Positives = `label == 1`.
* **Latency**: the text tower + mean pooling + L2 norm exported with `torch.onnx.export(dynamo=True)`
  (cosine to sentence-transformers = 1.0000). int8 via `onnxruntime.quantization.quantize_dynamic`.

### Caveats

* In **full** mode a passage is labeled only for the question whose candidate list contains it.
  Unlabeled passages count as non-relevant, so absolute scores are a lower bound. Configs are
  compared on identical qrels, so the comparisons stay fair.
* One run per config (deterministic exact search). The latency box is a 16 GB laptop running
  other apps, so treat ms numbers as relative.
* Corpus encoding ran on the M1 Pro GPU (MPS, fp32). Queries re-encoded on CPU give the same
  nDCG@10 to 4 decimals.

## Hardware & software

Apple M1 Pro (8-core CPU: 6P + 2E, 14-core GPU), 16 GB RAM, macOS 15.1.1. Python 3.12, torch 2.14.1,
transformers 5.20.1, sentence-transformers 6.1.0, faiss-cpu 1.15.1, onnxruntime 1.31.0.
Encoding wall time: EG2 95 min, EG1 71 min (MPS, batch 32).

## Reproduce

```bash
make install          # uv sync
make encode           # ~1.5 h per model on Apple-silicon GPU; resumable, cached in cache/
make latency          # exports ONNX fp32/int8 and benchmarks CPU query encoding
make reproduce        # BM25 + full grid + hybrid + tables + charts, from cached embeddings (~2 min)
```

or a single CLI: `uv run mrlqj {encode,bm25,evaluate,latency,report,reproduce}`. Everything is
driven by [`configs/grid.yaml`](configs/grid.yaml). [`notebooks/pareto.ipynb`](notebooks/pareto.ipynb)
redraws the chart from `results/grid.csv`.

```
mrl-quant-ja/
├── configs/grid.yaml          # models, dims, precisions, rescoring, eval modes
├── src/mrl_quant_ja/
│   ├── data.py                # JQaRA -> corpus / queries / qrels / candidates
│   ├── encode.py              # cache embeddings once at 768d fp32 (chunked, resumable)
│   ├── compress.py            # truncate + renorm + int8 / int8-dequantized / binary
│   ├── search.py              # flat IP / int8 / Hamming / rescore, rerank + full modes
│   ├── bm25_ja.py             # MeCab-tokenized BM25 (bm25s)
│   ├── evaluate.py            # ranx metrics, grid, RRF hybrid -> results/grid.csv
│   ├── report.py              # results/RESULTS.md + figures/*.png
│   └── cli.py                 # `mrlqj`
├── bench/latency.py           # torch vs ONNX fp32 vs ONNX int8 query encode
├── notebooks/pareto.ipynb
├── results/  figures/
└── Makefile
```
