"""`mrlqj` command line: encode -> evaluate -> bm25 -> latency -> report."""

from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG, load_config


def cmd_encode(cfg, args) -> None:
    from .data import load_jqara
    from .encode import encode_model

    data = load_jqara(cfg["dataset"]["name"], cfg["dataset"]["split"])
    for key in args.models or list(cfg["models"]):
        encode_model(cfg, key, data, device=args.device)


def cmd_bm25(cfg, args) -> None:
    from .bm25_ja import run_bm25

    run_bm25(cfg)


def cmd_evaluate(cfg, args) -> None:
    from .evaluate import run_all

    run_all(cfg)


def cmd_latency(cfg, args) -> None:
    import sys

    from .config import ROOT

    sys.path.insert(0, str(ROOT))
    from bench.latency import run_latency

    run_latency(cfg, skip_export=args.skip_export)


def cmd_report(cfg, args) -> None:
    from .report import build_report

    build_report(cfg)


def cmd_reproduce(cfg, args) -> None:
    """Everything downstream of the cached embeddings (no model forward passes)."""
    cmd_evaluate(cfg, args)
    cmd_report(cfg, args)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mrlqj", description=__doc__)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encode", help="encode JQaRA at 768d fp32 and cache")
    e.add_argument("--models", nargs="*", help="model keys from the config (default: all)")
    e.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    e.set_defaults(fn=cmd_encode)

    sub.add_parser("bm25", help="BM25 baseline with MeCab tokenization").set_defaults(fn=cmd_bm25)
    sub.add_parser("evaluate", help="dims x precisions grid + baselines + hybrid").set_defaults(fn=cmd_evaluate)
    lat = sub.add_parser("latency", help="CPU query-encode latency: torch fp32 vs ONNX int8")
    lat.add_argument("--skip-export", action="store_true", help="reuse exported ONNX files")
    lat.set_defaults(fn=cmd_latency)
    sub.add_parser("report", help="tables + Pareto chart").set_defaults(fn=cmd_report)
    sub.add_parser("reproduce", help="bm25 + grid + report from cached embeddings").set_defaults(fn=cmd_reproduce)

    args = p.parse_args(argv)
    args.fn(load_config(args.config), args)


if __name__ == "__main__":
    main()
