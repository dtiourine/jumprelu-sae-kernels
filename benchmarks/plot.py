import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _load(name):
    path = os.path.join(RESULTS, f"{name}.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def plot_sparsity():
    rows = _load("sweep")
    if not rows:
        print("no sweep.csv; skipping sparsity plot")
        return
    L0_rows = [r for r in rows if r["axis"] == "L0"]
    L0_rows.sort(key=lambda r: float(r["sparsity"]))
    x = [float(r["sparsity"]) for r in L0_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x,
        [float(r["speedup_kernel_exact"]) for r in L0_rows],
        "o-",
        label="exact (kernel)",
    )
    ax.plot(
        x,
        [float(r["speedup_kernel_fixed"]) for r in L0_rows],
        "s-",
        label="fixed (kernel)",
    )
    ax.axhline(1.0, color="k", ls="--", lw=1, label="dense parity")
    ax.set_xscale("log")
    ax.set_xlabel("sparsity (L0 / n_features)")
    ax.set_ylabel("speedup vs dense")
    ax.set_title("Sparse decode speedup vs sparsity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "speedup_vs_sparsity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_nfeatures_scaling():
    rows = _load("sweep")
    if not rows:
        return
    nf = [r for r in rows if r["axis"] == "n_features"]
    nf.sort(key=lambda r: int(r["n_features"]))
    x = [int(r["n_features"]) for r in nf]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, [float(r["dense_ms"]) for r in nf], "o-", label="dense")
    ax.plot(x, [float(r["kernel_exact_ms"]) for r in nf], "s-", label="exact (kernel)")
    ax.set_xlabel("n_features")
    ax.set_ylabel("time (ms)")  # noqa: E702
    ax.set_title("Decode time vs dictionary size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "scaling_vs_nfeatures.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_memory():
    rows = _load("memory")
    if not rows:
        print("no memory.csv; skipping memory plot")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for B in sorted({int(r["B"]) for r in rows}):
        sub = [r for r in rows if int(r["B"]) == B]
        sub.sort(key=lambda r: int(r["max_l0"]))
        ax.plot(
            [int(r["max_l0"]) for r in sub],
            [float(r["fixed_bytes"]) / 1e6 for r in sub],
            "o-",
            label=f"fixed B={B}",
        )
    ax.set_xlabel("max_l0")
    ax.set_ylabel("peak memory (MB)")
    ax.set_title("Fixed-variant memory vs max_l0")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "memory_vs_maxl0.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_full_vs_kernel():
    rows = _load("sweep")
    if not rows:
        return
    L0_rows = [r for r in rows if r["axis"] == "L0"]
    L0_rows.sort(key=lambda r: float(r["sparsity"]))
    x = [float(r["sparsity"]) for r in L0_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        x,
        [float(r["full_exact_ms"]) for r in L0_rows],
        "o-",
        label="full pipeline (build + decode)",
    )
    ax.plot(
        x,
        [float(r["kernel_exact_ms"]) for r in L0_rows],
        "s-",
        label="decode kernel only",
    )
    ax.set_xscale("log")
    ax.set_xlabel("sparsity (L0 / n_features)")
    ax.set_ylabel("time (ms)")
    ax.set_title("Full pipeline vs decode-only (gap = CSR construction)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "full_vs_kernel.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_validate_overhead():
    """fixed full-pipeline speedup with vs without the validate sync, along the
    d_model axis. The sync is a ~constant cost, so the gap closes as the decode
    grows — i.e. the safety check is cheap once the decode is non-trivial."""
    rows = _load("sweep")
    if not rows:
        return
    if "speedup_full_fixed_validate" not in rows[0]:
        print("sweep.csv has no validate columns; rerun bench_sweep — skipping")
        return
    dm = [r for r in rows if r["axis"] == "d_model"]
    dm.sort(key=lambda r: int(r["d_model"]))
    x = [int(r["d_model"]) for r in dm]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, [float(r["speedup_full_fixed"]) for r in dm], "o-",
            label="fixed (validate=False)")
    ax.plot(x, [float(r["speedup_full_fixed_validate"]) for r in dm], "s-",
            label="fixed+validate (validate=True)")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="dense parity")
    ax.set_xlabel("d_model")
    ax.set_ylabel("full-pipeline speedup vs dense")
    ax.set_title("Validate sync cost vs decode size (gap closes as d_model grows)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(RESULTS, "validate_overhead.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_baselines_vs_spmm():
    """Grouped bars: our variants' full-pipeline speedup vs cuSPARSE
    (torch.sparse.mm) across the baseline points. >1 means faster than cuSPARSE."""
    rows = _load("baselines")
    if not rows:
        print("no baselines.csv; skipping vs-cuSPARSE plot")
        return
    if "full_fixed_validate_vs_spmm" not in rows[0]:
        print("baselines.csv has no fixed-vs-spmm columns; rerun bench_baselines — skipping")
        return
    labels = [f"B{r['B']} F{r['n_features']}\nD{r['d_model']} L{r['L0']}" for r in rows]
    series = [
        ("exact", "full_ours_vs_spmm"),
        ("fixed", "full_fixed_vs_spmm"),
        ("fixed+validate", "full_fixed_validate_vs_spmm"),
    ]
    n, width = len(rows), 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, col) in enumerate(series):
        xs = [j + (i - 1) * width for j in range(n)]
        ax.bar(xs, [float(r[col]) for r in rows], width, label=name)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="cuSPARSE parity")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("full-pipeline speedup vs cuSPARSE")
    ax.set_title("Our variants vs cuSPARSE (torch.sparse.mm)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    out = os.path.join(RESULTS, "baselines_vs_spmm.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    plot_sparsity()
    plot_nfeatures_scaling()
    plot_memory()
    plot_full_vs_kernel()
    plot_validate_overhead()
    plot_baselines_vs_spmm()


if __name__ == "__main__":
    main()
