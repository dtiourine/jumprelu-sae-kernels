import torch

from benchmarks.lib.harness import capture_env, bench, write_results
from benchmarks.lib.data import fixed_l0_feature_acts
from jumprelu_sae_kernels import sparse_decode
from jumprelu_sae_kernels.exact.wrappers import (
    build_csr as build_csr_exact,
    _sparse_decode as decode_exact,
)
from jumprelu_sae_kernels.fixed.wrappers import (
    build_csr as build_csr_fixed,
    _sparse_decode as decode_fixed,
)

BASE = dict(B=32, n_features=65536, d_model=768, L0=64)

AXES = {
    "L0": [16, 32, 64, 128, 256, 512, 1024, 4096, 16384],
    "B": [1, 4, 16, 64, 256, 1024, 4096],
    "d_model": [256, 512, 768, 1024, 2048, 4096],
    "n_features": [4096, 16384, 32768, 65536, 131072],
}


def _bench_point(B, n_features, d_model, L0):
    W = torch.randn(n_features, d_model, device="cuda").contiguous()
    acts = fixed_l0_feature_acts(B, n_features, min(L0, n_features))
    max_l0 = max(min(L0, n_features), 1)

    dense = bench(lambda: acts @ W)["median_ms"]
    full_exact = bench(lambda: sparse_decode(acts, W, variant="exact"))["median_ms"]
    full_fixed = bench(lambda: sparse_decode(acts, W, variant="fixed", max_l0=max_l0))[
        "median_ms"
    ]

    fi, fv, ro, B_ = build_csr_exact(acts)
    kern_exact = bench(lambda: decode_exact(fi, fv, ro, W, B_))["median_ms"]
    fi2, fv2, c2, B2, ml = build_csr_fixed(acts, max_l0=max_l0)
    kern_fixed = bench(lambda: decode_fixed(fi2, fv2, c2, W, B2, ml))["median_ms"]

    return dict(
        B=B,
        n_features=n_features,
        d_model=d_model,
        L0=L0,
        sparsity=min(L0, n_features) / n_features,
        dense_ms=dense,
        full_exact_ms=full_exact,
        kernel_exact_ms=kern_exact,
        full_fixed_ms=full_fixed,
        kernel_fixed_ms=kern_fixed,
        speedup_full_exact=dense / full_exact,
        speedup_kernel_exact=dense / kern_exact,
        speedup_full_fixed=dense / full_fixed,
        speedup_kernel_fixed=dense / kern_fixed,
    )


def main():
    env = capture_env()
    rows = []
    for axis, values in AXES.items():
        for v in values:
            cfg = dict(BASE)
            cfg[axis] = v
            row = _bench_point(**cfg)
            row["axis"] = axis
            rows.append(row)
            print(
                f"{axis}={v}: dense={row['dense_ms']:.3f} "
                f"kern_exact={row['kernel_exact_ms']:.3f} "
                f"({row['speedup_kernel_exact']:.2f}x) "
                f"kern_fixed={row['kernel_fixed_ms']:.3f} "
                f"({row['speedup_kernel_fixed']:.2f}x)"
            )
    csv_path, _ = write_results(rows, "sweep", env)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
