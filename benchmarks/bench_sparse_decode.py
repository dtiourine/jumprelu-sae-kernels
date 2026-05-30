"""Benchmark the sparse JumpReLU decoder against dense matmul (cuBLAS).

Measures the wall-clock speedup of  sparse_decode(feature_acts, W_dec)
versus the dense baseline  feature_acts @ W_dec, swept across sparsity
ratios (L0 / n_features) at realistic SAE shapes.

The key output is the *crossover*: above what sparsity does the sparse
kernel beat cuBLAS? Real SAEs run at L0/n_features ~ 0.1-1%, deep in the
regime where the sparse kernel should win.

Run:  python benchmarks/bench_decode.py
"""

import torch
import triton

from kernel_jumprelu_sae.wrapper import sparse_decode


DEVICE = "cuda"


def make_sparse_acts(B, n_features, L0, seed=0):
    """[B, n_features] activations with exactly L0 active features per token."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    acts = torch.zeros(B, n_features, device=DEVICE)
    for b in range(B):
        fired = torch.randperm(n_features, generator=g, device=DEVICE)[:L0]
        acts[b, fired] = torch.randn(L0, generator=g, device=DEVICE)
    return acts


def bench_one(B, n_features, d_model, L0):
    """Return (dense_ms, sparse_ms) for one configuration."""
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, L0)

    # do_bench handles warmup, multiple runs, and CUDA synchronization
    dense_ms = triton.testing.do_bench(lambda: acts @ W_dec)
    sparse_ms = triton.testing.do_bench(lambda: sparse_decode(acts, W_dec))
    return dense_ms, sparse_ms


def main():
    B = 32
    n_features = 65536          # realistic large SAE dictionary
    d_model = 768               # e.g. GPT-2 residual width

    # sweep L0 from very sparse (realistic) to dense (where we expect to lose)
    L0_values = [32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]

    print(f"B={B}, n_features={n_features}, d_model={d_model}")
    print(f"{'L0':>7} {'sparsity':>10} {'dense_ms':>10} {'sparse_ms':>11} {'speedup':>9}")
    print("-" * 52)

    for L0 in L0_values:
        dense_ms, sparse_ms = bench_one(B, n_features, d_model, L0)
        sparsity = L0 / n_features
        speedup = dense_ms / sparse_ms
        marker = "  <-- win" if speedup > 1 else ""
        print(f"{L0:>7} {sparsity:>9.2%} {dense_ms:>10.3f} "
              f"{sparse_ms:>11.3f} {speedup:>8.2f}x{marker}")


if __name__ == "__main__":
    main()