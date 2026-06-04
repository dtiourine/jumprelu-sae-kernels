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

POINTS = [
    dict(B=32, n_features=65536, d_model=768, L0=64),
    dict(B=256, n_features=65536, d_model=768, L0=64),
    dict(B=32, n_features=131072, d_model=512, L0=128),
]


def _gather_matmul(acts, W):
    """Strawman: per-token gather of fired rows then small matmul. Loops over
    tokens (ragged), so this is intentionally the naive approach."""
    B, d_model = acts.shape[0], W.shape[1]
    out = torch.zeros(B, d_model, device=acts.device, dtype=torch.float32)
    for b in range(B):
        nz = acts[b].nonzero().squeeze(-1)
        if nz.numel():
            out[b] = acts[b, nz].float() @ W[nz].float()
    return out


def main():
    env = capture_env()
    rows = []
    for p in POINTS:
        B, F, D, L0 = p["B"], p["n_features"], p["d_model"], p["L0"]
        W = torch.randn(F, D, device="cuda").contiguous()
        acts = fixed_l0_feature_acts(B, F, L0)

        # Full pipeline from a dense activation tensor (construction + decode).
        dense = bench(lambda: acts @ W)["median_ms"]
        gather = bench(lambda: _gather_matmul(acts, W))["median_ms"]
        ours_full = bench(lambda: sparse_decode(acts, W, variant="exact"))["median_ms"]
        ours_full_fixed = bench(
            lambda: sparse_decode(acts, W, variant="fixed", max_l0=L0, validate=False)
        )["median_ms"]
        ours_full_fixed_val = bench(
            lambda: sparse_decode(acts, W, variant="fixed", max_l0=L0, validate=True)
        )["median_ms"]
        spmm_full = bench(lambda: torch.sparse.mm(acts.to_sparse_csr(), W))["median_ms"]

        # Decode only (layout pre-built outside the timed region). 
        fi, fv, ro, B_ = build_csr_exact(acts)
        ours_decode = bench(lambda: decode_exact(fi, fv, ro, W, B_))["median_ms"]
        fi2, fv2, c2, B2, ml = build_csr_fixed(acts, max_l0=L0, validate=False)
        ours_decode_fixed = bench(lambda: decode_fixed(fi2, fv2, c2, W, B2, ml))["median_ms"]
        sp = acts.to_sparse_csr()
        spmm_decode = bench(lambda: torch.sparse.mm(sp, W))["median_ms"]

        row = dict(
            **p,
            dense_ms=dense,
            gather_matmul_ms=gather,
            ours_full_ms=ours_full,
            ours_full_fixed_ms=ours_full_fixed,
            ours_full_fixed_validate_ms=ours_full_fixed_val,
            spmm_full_ms=spmm_full,
            ours_decode_ms=ours_decode,
            ours_decode_fixed_ms=ours_decode_fixed,
            spmm_decode_ms=spmm_decode,
            full_speedup_vs_dense=dense / ours_full,
            full_fixed_speedup_vs_dense=dense / ours_full_fixed,
            full_fixed_validate_speedup_vs_dense=dense / ours_full_fixed_val,
            full_ours_vs_spmm=spmm_full / ours_full,
            full_fixed_vs_spmm=spmm_full / ours_full_fixed,
            full_fixed_validate_vs_spmm=spmm_full / ours_full_fixed_val,
            decode_ours_vs_spmm=spmm_decode / ours_decode,
            decode_fixed_vs_spmm=spmm_decode / ours_decode_fixed,
            construction_overhead_ms=ours_full - ours_decode,
            validate_overhead_ms=ours_full_fixed_val - ours_full_fixed,
        )
        rows.append(row)
        print(f"B{B} F{F} D{D} L{L0}:")
        print(f"  full:   dense={dense:.3f} gather={gather:.3f} "
              f"exact={ours_full:.3f} fixed={ours_full_fixed:.3f} "
              f"fixed+val={ours_full_fixed_val:.3f} spmm={spmm_full:.3f}")
        print(f"          exact vs dense {dense / ours_full:.2f}x | "
              f"exact vs spmm {spmm_full / ours_full:.2f}x | "
              f"fixed vs spmm {spmm_full / ours_full_fixed:.2f}x | "
              f"fixed+val vs spmm {spmm_full / ours_full_fixed_val:.2f}x")
        print(f"  decode: exact={ours_decode:.3f} fixed={ours_decode_fixed:.3f} "
              f"spmm={spmm_decode:.3f} "
              f"(exact vs spmm {spmm_decode / ours_decode:.2f}x | "
              f"fixed vs spmm {spmm_decode / ours_decode_fixed:.2f}x)")
        print(f"  CSR-build overhead (exact full): {ours_full - ours_decode:.3f} ms | "
              f"validate sync overhead: {ours_full_fixed_val - ours_full_fixed:.3f} ms")
    csv_path, _ = write_results(rows, "baselines", env)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
