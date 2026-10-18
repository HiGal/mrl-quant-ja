.PHONY: install encode bm25 evaluate latency report reproduce all clean-results

install:
	uv sync

# Expensive: one model pass per model (~1.5-2 h each on an M1 Pro GPU). Resumable.
encode:
	uv run mrlqj encode

bm25:
	uv run mrlqj bm25

evaluate:
	uv run mrlqj evaluate

# CPU query-encode latency: PyTorch fp32 vs ONNX Runtime fp32 / int8 (exports ONNX once).
latency:
	uv run mrlqj latency

report:
	uv run mrlqj report

# Regenerate every table and chart from cached embeddings (no model forward passes).
reproduce:
	uv run mrlqj reproduce

all: encode latency reproduce

clean-results:
	rm -f results/*.csv results/RESULTS.md figures/*.png
