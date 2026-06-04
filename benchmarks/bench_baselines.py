"""Compare the kernel against alternative baselines, fairly separating the
CSR-construction cost from the decode cost.

For each point we time two regimes:
  * full pipeline — starting from a dense [B, n_features] activation tensor (what
    a user actually has post-encoder): construction + decode, for both our kernel
    (build_csr) and torch.sparse (to_sparse_csr), vs dense cuBLAS and naive
    gather+matmul.
  * decode only — CSR/sparse layout pre-built OUTSIDE the timed region: our
    decode kernel vs torch.sparse.mm, isolating kernel-vs-cuSPARSE.

The separation matters: our decode kernel is competitive with cuSPARSE, but the
CSR-construction pass dominates our full pipeline — so construction, not the
decode, is the optimization target. Timing the full pipeline against a
pre-built spmm would be apples-to-oranges.

Run:  uv run python -m benchmarks.bench_baselines
Writes benchmarks/results/baselines.{csv,json}.
"""

import torch

from benchmarks.lib.harness import capture_env, bench, write_results
from benchmarks.lib.data import fixed_l0_feature_acts
from jumprelu_sae_kernels import sparse_decode
from jumprelu_sae_kernels.exact.wrappers import build_csr, _sparse_decode

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

        # --- full pipeline: from a dense activation tensor (construction + decode)
        dense = bench(lambda: acts @ W)["median_ms"]
        gather = bench(lambda: _gather_matmul(acts, W))["median_ms"]
        ours_full = bench(lambda: sparse_decode(acts, W, variant="exact"))["median_ms"]
        spmm_full = bench(lambda: torch.sparse.mm(acts.to_sparse_csr(), W))["median_ms"]

        # --- decode only: layout pre-built outside the timed region
        fi, fv, ro, B_ = build_csr(acts)
        ours_decode = bench(lambda: _sparse_decode(fi, fv, ro, W, B_))["median_ms"]
        sp = acts.to_sparse_csr()
        spmm_decode = bench(lambda: torch.sparse.mm(sp, W))["median_ms"]

        row = dict(
            **p,
            dense_ms=dense,
            gather_matmul_ms=gather,
            ours_full_ms=ours_full,
            spmm_full_ms=spmm_full,
            ours_decode_ms=ours_decode,
            spmm_decode_ms=spmm_decode,
            full_speedup_vs_dense=dense / ours_full,
            full_ours_vs_spmm=spmm_full / ours_full,
            decode_ours_vs_spmm=spmm_decode / ours_decode,
            construction_overhead_ms=ours_full - ours_decode,
        )
        rows.append(row)
        print(f"B{B} F{F} D{D} L{L0}:")
        print(f"  full:   dense={dense:.3f} gather={gather:.3f} "
              f"ours={ours_full:.3f} spmm={spmm_full:.3f} "
              f"(ours vs dense {dense / ours_full:.2f}x, ours vs spmm {spmm_full / ours_full:.2f}x)")
        print(f"  decode: ours={ours_decode:.3f} spmm={spmm_decode:.3f} "
              f"(ours vs spmm {spmm_decode / ours_decode:.2f}x)")
        print(f"  CSR-build overhead in our full pipeline: "
              f"{ours_full - ours_decode:.3f} ms")
    csv_path, _ = write_results(rows, "baselines", env)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
