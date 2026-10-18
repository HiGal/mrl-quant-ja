"""Turn results/*.csv into results/RESULTS.md tables and figures/pareto.png."""

from __future__ import annotations

import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .compress import bytes_per_vector

# Reference categorical palette (light mode), fixed order: one hue per precision.
# Three hues (scatter all-pairs cap); the two int8 scorings share orange, told apart by fill.
COLORS = {"float32": "#2a78d6", "int8": "#eb6834", "int8_deq": "#eb6834", "ubinary": "#1baf7a"}
HOLLOW = {"int8"}
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
PREC_LABEL = {"float32": "float32", "int8": "int8 (code·code)", "int8_deq": "int8 (dequantized)",
              "ubinary": "binary"}
KNEE_RETENTION = 0.95  # knee = smallest index that keeps >= 95% of 768d fp32 nDCG@10


def config_name(r) -> str:
    if r["precision"] == "sparse":
        return "—"
    name = f"{int(r['dim'])}d {PREC_LABEL[r['precision']]}"
    if isinstance(r["rescore"], str):
        name += f" → {PREC_LABEL[r['rescore']].split(' (')[0]} rescore"
    return name


def _wide(df: pd.DataFrame) -> pd.DataFrame:
    """One row per config, metric columns suffixed with the eval mode."""
    keys = ["model", "label", "dim", "precision", "rescore", "bytes_per_vector"]
    df = df.copy()
    sentinel = {"dim": -1, "rescore": "", "bytes_per_vector": -1}
    df = df.fillna(sentinel)
    w = df.set_index(keys + ["mode"])[["ndcg@10", "recall@100", "mrr@10"]].unstack("mode")
    w.columns = [f"{m} {mode}" for m, mode in w.columns]
    w = w.reset_index()
    for col, v in sentinel.items():
        w[col] = w[col].replace(v, np.nan)
    return w


def disk_bytes(r) -> float:
    """Searched index + the representation kept on disk for rescoring."""
    extra = bytes_per_vector(int(r["dim"]), r["rescore"]) if isinstance(r["rescore"], str) else 0
    return r["bytes_per_vector"] + extra


def find_knee(eg2: pd.DataFrame, metric: str = "ndcg@10 full") -> pd.Series:
    """Smallest in-memory index within KNEE_RETENTION of 768d fp32; ties -> less disk."""
    ref = eg2[(eg2.dim == 768) & (eg2.precision == "float32") & eg2.rescore.isna()][metric].iloc[0]
    ok = eg2[eg2[metric] >= KNEE_RETENTION * ref].copy()
    ok["_disk"] = ok.apply(disk_bytes, axis=1)
    return ok.sort_values(["bytes_per_vector", "_disk"]).iloc[0]


def pareto_front(points: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    pts = points.sort_values([x, y], ascending=[True, False])
    front, best = [], -np.inf
    for _, r in pts.iterrows():
        if r[y] > best:
            front.append(r)
            best = r[y]
    return pd.DataFrame(front)


def fmt_table(df: pd.DataFrame, cols: dict[str, str], floatfmt: str = ".4f") -> str:
    return df[list(cols)].rename(columns=cols).to_markdown(index=False, floatfmt=floatfmt)


def plot_pareto(w: pd.DataFrame, knee: pd.Series, out, mode: str = "full") -> None:
    y = f"ndcg@10 {mode}"
    eg2 = w[w.model == "eg2"].copy()
    eg2["mb"] = eg2.bytes_per_vector  # MB per 1M docs == bytes per vector

    plt.rcParams.update({"font.family": ["Hiragino Sans", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    front = pareto_front(eg2.dropna(subset=[y]), "mb", y)
    ax.step(front.mb, front[y], where="post", color=MUTED, lw=1.2, ls="--", zorder=1,
            label="Pareto frontier")

    for prec, color in COLORS.items():
        sub = eg2[(eg2.precision == prec) & eg2.rescore.isna()]
        hollow = prec in HOLLOW
        ax.scatter(sub.mb, sub[y], s=56 if hollow else 64, facecolor="#fcfcfb" if hollow else color,
                   edgecolor=color if hollow else "#fcfcfb", lw=1.8 if hollow else 2, zorder=3,
                   label=PREC_LABEL[prec])
        for _, r in sub.iterrows():
            ax.annotate(f"{int(r.dim)}", (r.mb, r[y]), xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=INK_2)
    resc = eg2[eg2.rescore.notna()]
    if len(resc) and resc[y].notna().any():
        for marker, (rname, sub) in zip(("D", "s"), resc.groupby("rescore")):
            ax.scatter(sub.mb, sub[y], s=52, marker=marker, facecolor="#fcfcfb",
                       edgecolor=COLORS["ubinary"], lw=1.8, zorder=3,
                       label=f"binary → {PREC_LABEL[rname].split(' (')[0]} rescore of top-100")

    eg1 = w[(w.model == "eg1") & (w.dim == 768) & (w.precision == "float32") & w.rescore.isna()]
    if len(eg1):
        ax.scatter(eg1.bytes_per_vector, eg1[y], s=90, marker="*", color=INK_2, zorder=3,
                   label="EmbeddingGemma 1 · 768d float32")
    bm25 = w[w.model == "bm25"]
    if len(bm25):
        ax.axhline(bm25[y].iloc[0], color=INK_2, lw=1, ls=":", zorder=1)
        ax.text(0.01, bm25[y].iloc[0], "BM25 (MeCab)", va="bottom", fontsize=8, color=INK_2,
                transform=ax.get_yaxis_transform())

    if mode == "full":
        ax.scatter([knee.bytes_per_vector], [knee[y]], s=260, facecolor="none", edgecolor=INK,
                   lw=1.2, zorder=4)
        ax.annotate(f"knee: {config_name(knee)}\n{knee.bytes_per_vector:g} B/vector, "
                    f"nDCG@10 {knee[y]:.3f}", (knee.bytes_per_vector, knee[y]),
                    xytext=(8, -185), textcoords="offset points", fontsize=8.5, color=INK,
                    arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8})

    ax.set_xscale("log", base=2)
    ticks = sorted(eg2.mb.unique())
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Index size, MB per 1M documents (log scale)", color=INK_2)
    ax.set_ylabel("nDCG@10", color=INK_2)
    title = {"full": "JQaRA full-corpus retrieval (144k passages)",
             "rerank": "JQaRA official protocol (rerank 100 candidates)"}[mode]
    ax.set_title(f"How small can a Japanese search vector get?\n{title}", loc="left",
                 color=INK, fontsize=11.5)
    ax.grid(True, color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=3, fontsize=8, frameon=False,
              labelcolor=INK_2, title="EmbeddingGemma 2 (point labels = dimensions)",
              title_fontsize=8, alignment="left")
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_model_compare(w: pd.DataFrame, out, y: str = "ndcg@10 full") -> None:
    """EmbeddingGemma 2 vs 1 across Matryoshka dims, for two storage recipes."""
    plt.rcParams.update({"font.family": ["Hiragino Sans", "DejaVu Sans"], "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    models = {"eg2": ("EmbeddingGemma 2", "#2a78d6"), "eg1": ("EmbeddingGemma 1", "#eb6834")}
    recipes = {("float32", None): ("float32", "-", "o"),
               ("ubinary", "int8_deq"): ("binary → int8 rescore", "--", "s")}
    for key, (mlabel, color) in models.items():
        for (prec, resc), (rlabel, ls, marker) in recipes.items():
            sub = w[(w.model == key) & (w.precision == prec)]
            sub = sub[sub.rescore.isna()] if resc is None else sub[sub.rescore == resc]
            sub = sub.sort_values("dim")
            if not len(sub):
                continue
            ax.plot(sub.dim, sub[y], ls=ls, color=color, lw=2, marker=marker, ms=7,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=f"{mlabel} · {rlabel}")
    bm25 = w[w.model == "bm25"]
    if len(bm25):
        ax.axhline(bm25[y].iloc[0], color=INK_2, lw=1, ls=":")
        ax.text(0.01, bm25[y].iloc[0], "BM25 (MeCab)", va="bottom", fontsize=8, color=INK_2,
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512, 768], ["128", "256", "512", "768"])
    ax.minorticks_off()
    ax.set_xlabel("Matryoshka dimensions", color=INK_2)
    ax.set_ylabel("nDCG@10 (full corpus)", color=INK_2)
    ax.set_title("EmbeddingGemma 2 vs 1 on Japanese retrieval (JQaRA)", loc="left", color=INK, fontsize=11.5)
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)
    ax.legend(loc="lower right", fontsize=8, frameon=False, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_report(cfg: dict[str, Any]) -> None:
    res, figs = cfg["results_dir"], cfg["figures_dir"]
    df = pd.read_csv(res / "grid.csv")
    w = _wide(df)
    eg2 = w[w.model == "eg2"].copy()
    ref = eg2[(eg2.dim == 768) & (eg2.precision == "float32") & eg2.rescore.isna()].iloc[0]
    eg2["compression"] = ref.bytes_per_vector / eg2.bytes_per_vector
    eg2["retention"] = eg2["ndcg@10 full"] / ref["ndcg@10 full"] * 100
    eg2["config"] = eg2.apply(config_name, axis=1)
    eg2["disk"] = eg2.apply(disk_bytes, axis=1)
    order = {"float32": 0, "int8": 1, "int8_deq": 2, "ubinary": 3}
    eg2 = eg2.sort_values(["dim", "precision", "rescore"], key=lambda s: s.map(order) if s.name == "precision"
                          else (-s if s.name == "dim" else s.fillna("")))
    knee = find_knee(eg2)

    plot_pareto(w, knee, figs / "pareto.png", "full")
    plot_pareto(w, knee, figs / "pareto_rerank.png", "rerank")
    plot_model_compare(w, figs / "eg2_vs_eg1.png")

    lines = ["# Results", "",
             f"Dataset: `{cfg['dataset']['name']}` ({cfg['dataset']['split']}), 1,667 queries.",
             "*full* = exact top-100 retrieval over all 144,372 test passages; "
             "*rerank* = official JQaRA protocol (rank each query's 100 candidates).", "",
             "## EmbeddingGemma 2: Matryoshka × quantization grid", "",
             fmt_table(eg2, {"config": "config", "bytes_per_vector": "bytes/vec (RAM)",
                             "disk": "bytes/vec (disk)",
                             "compression": "compression ×", "ndcg@10 full": "nDCG@10 (full)",
                             "recall@100 full": "R@100 (full)", "mrr@10 full": "MRR@10 (full)",
                             "ndcg@10 rerank": "nDCG@10 (rerank)", "retention": "% of 768d fp32"},
                       floatfmt=".4g"),
             "", f"**Knee** (smallest index keeping ≥{KNEE_RETENTION:.0%} of 768d fp32 full-corpus nDCG@10): "
             f"**{config_name(knee)}** — {knee.bytes_per_vector:g} bytes/vector in RAM, "
             f"{disk_bytes(knee):g} on disk "
             f"({ref.bytes_per_vector / knee.bytes_per_vector:.0f}× smaller), "
             f"nDCG@10 {knee['ndcg@10 full']:.4f} vs {ref['ndcg@10 full']:.4f}.", ""]

    base = w[w.model != "eg2"].copy()
    base = base[~((base.model == "eg1") & ~((base.dim == 768) & (base.precision == "float32")
                                            & base.rescore.isna()))]
    base = pd.concat([base, ref.to_frame().T])
    base["config"] = base.apply(config_name, axis=1)
    rank = {"bm25": 0, "eg1": 1, "eg2": 2}
    base = base.sort_values(["model", "bytes_per_vector"], key=lambda s: s.map(lambda m: rank.get(m, 3))
                            if s.name == "model" else s)
    lines += ["## Baselines", "",
              fmt_table(base, {"label": "system", "config": "config", "ndcg@10 full": "nDCG@10 (full)",
                               "recall@100 full": "R@100 (full)", "ndcg@10 rerank": "nDCG@10 (rerank)",
                               "mrr@10 rerank": "MRR@10 (rerank)"}), ""]

    eg1 = w[w.model == "eg1"].copy()
    if len(eg1) > 1:
        eg1["config"] = eg1.apply(config_name, axis=1)
        eg1 = eg1.sort_values(["dim", "precision"], ascending=[False, True])
        lines += ["## EmbeddingGemma 1: same grid (for comparison)", "",
                  fmt_table(eg1, {"config": "config", "ndcg@10 full": "nDCG@10 (full)",
                                  "recall@100 full": "R@100 (full)", "ndcg@10 rerank": "nDCG@10 (rerank)"}),
                  ""]

    if (res / "latency.csv").exists():
        lat = pd.read_csv(res / "latency.csv")
        hw = json.loads((res / "hardware.json").read_text()) if (res / "hardware.json").exists() else {}
        lines += ["## CPU query-encode latency", "",
                  f"{hw.get('cpu', '')}, {hw.get('threads', '')} threads, {hw.get('ram_gb', '')} GB RAM; "
                  "end-to-end incl. tokenization; real JQaRA queries.", "",
                  fmt_table(lat, {"backend": "backend", "batch_size": "batch", "model_mb": "model MB",
                                  "batch_ms_p50": "p50 ms/batch", "batch_ms_p95": "p95 ms/batch",
                                  "ms_per_query": "ms/query", "queries_per_s": "queries/s"}, floatfmt=".1f"),
                  ""]
    if (res / "onnx_fidelity.csv").exists():
        fid = pd.read_csv(res / "onnx_fidelity.csv")
        lines += ["### Does an int8 query encoder hurt retrieval?", "",
                  "All 1,667 queries re-encoded on CPU by each backend, scored against the same corpus.", "",
                  fid.to_markdown(index=False, floatfmt=".4f"), ""]

    lines += ["![Pareto](../figures/pareto.png)", ""]
    (res / "RESULTS.md").write_text("\n".join(lines))
    print(f"[report] wrote {res / 'RESULTS.md'}, {figs / 'pareto.png'}")
    print(f"[report] knee: {config_name(knee)} nDCG@10={knee['ndcg@10 full']:.4f}")
