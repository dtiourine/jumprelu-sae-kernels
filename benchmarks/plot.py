"""Render figures from benchmarks/results/*.csv. CPU-only; no GPU or kernels
needed, so figures regenerate anywhere the CSVs are present.

Run:  uv run python -m benchmarks.plot
Writes PNGs into benchmarks/results/.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    ax.plot(x, [float(r["speedup_kernel_exact"]) for r in L0_rows], "o-", label="exact (kernel)")
    ax.plot(x, [float(r["speedup_kernel_fixed"]) for r in L0_rows], "s-", label="fixed (kernel)")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="dense parity")
    ax.set_xscale("log"); ax.set_xlabel("sparsity (L0 / n_features)")  # noqa: E702
    ax.set_ylabel("speedup vs dense"); ax.set_title("Sparse decode speedup vs sparsity")  # noqa: E702
    ax.legend(); ax.grid(True, alpha=0.3)  # noqa: E702
    out = os.path.join(RESULTS, "speedup_vs_sparsity.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)  # noqa: E702
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
    ax.set_xlabel("n_features"); ax.set_ylabel("time (ms)")  # noqa: E702
    ax.set_title("Decode time vs dictionary size"); ax.legend(); ax.grid(True, alpha=0.3)  # noqa: E702
    out = os.path.join(RESULTS, "scaling_vs_nfeatures.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)  # noqa: E702
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
        ax.plot([int(r["max_l0"]) for r in sub],
                [float(r["fixed_bytes"]) / 1e6 for r in sub], "o-", label=f"fixed B={B}")
    ax.set_xlabel("max_l0"); ax.set_ylabel("peak memory (MB)")  # noqa: E702
    ax.set_title("Fixed-variant memory vs max_l0"); ax.legend(); ax.grid(True, alpha=0.3)  # noqa: E702
    out = os.path.join(RESULTS, "memory_vs_maxl0.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)  # noqa: E702
    print(f"wrote {out}")


def plot_full_vs_kernel():
    """Full pipeline vs decode-only along the L0 axis: the gap is the
    CSR-construction overhead (the optimization target)."""
    rows = _load("sweep")
    if not rows:
        return
    L0_rows = [r for r in rows if r["axis"] == "L0"]
    L0_rows.sort(key=lambda r: float(r["sparsity"]))
    x = [float(r["sparsity"]) for r in L0_rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, [float(r["full_exact_ms"]) for r in L0_rows], "o-", label="full pipeline (build + decode)")
    ax.plot(x, [float(r["kernel_exact_ms"]) for r in L0_rows], "s-", label="decode kernel only")
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


def main():
    plot_sparsity()
    plot_nfeatures_scaling()
    plot_memory()
    plot_full_vs_kernel()


if __name__ == "__main__":
    main()
