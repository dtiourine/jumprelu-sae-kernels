"""Benchmark the sparse JumpReLU decoder against dense matmul (cuBLAS).

Measures wall-clock time across sparsity ratios (L0 / n_features) at
realistic SAE shapes, separating two costs:

  full_ms   = sparse_decode(...)         end-to-end: CSR build + kernel
  kernel_ms = _sparse_decode(...)        kernel only (CSR built beforehand)

Comparing full vs kernel localizes where time goes: if kernel_ms << full_ms,
the nonzero/CSR preprocessing dominates (motivating a fused extraction
kernel); if kernel_ms ~ full_ms, the kernel itself is the bottleneck
(motivating memory/coalescing optimization).

Baseline is dense  feature_acts @ W_dec  (routes to cuBLAS).

Run:  python benchmarks/bench_sparse_decode.py
"""

import torch
import triton

from kernel_jumprelu_sae.exact.wrappers import build_csr, _sparse_decode, sparse_decode


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
    """Return (dense_ms, full_ms, kernel_ms) for one configuration."""
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, L0)

    # dense baseline (cuBLAS)
    dense_ms = triton.testing.do_bench(lambda: acts @ W_dec)

    # full pipeline: CSR construction + kernel, as a user pays for it
    full_ms = triton.testing.do_bench(lambda: sparse_decode(acts, W_dec))

    # kernel only: build CSR OUTSIDE the timed region to isolate the kernel
    Wc = W_dec.contiguous()
    flat_idx, flat_val, row_offsets, B_ = build_csr(acts)
    kernel_ms = triton.testing.do_bench(
        lambda: _sparse_decode(flat_idx, flat_val, row_offsets, Wc, B_)
    )

    return dense_ms, full_ms, kernel_ms


def main():
    B = 32
    n_features = 65536          # realistic large SAE dictionary
    d_model = 768               # e.g. GPT-2 residual width

    # sweep L0 from very sparse (realistic) to dense (where we expect to lose)
    L0_values = [32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]

    print(f"B={B}, n_features={n_features}, d_model={d_model}")
    print(f"{'L0':>7} {'sparsity':>9} {'dense':>9} {'full':>9} {'kernel':>9} "
          f"{'full_sp':>9} {'kern_sp':>9}")
    print("-" * 70)

    for L0 in L0_values:
        dense_ms, full_ms, kernel_ms = bench_one(B, n_features, d_model, L0)
        sparsity = L0 / n_features
        full_sp = dense_ms / full_ms          # speedup of full pipeline vs dense
        kern_sp = dense_ms / kernel_ms         # speedup of kernel alone vs dense
        print(f"{L0:>7} {sparsity:>8.2%} {dense_ms:>9.3f} {full_ms:>9.3f} "
              f"{kernel_ms:>9.3f} {full_sp:>8.2f}x {kern_sp:>8.2f}x")


if __name__ == "__main__":
    main()