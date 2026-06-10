import torch

from benchmarks.lib.harness import capture_env, bench, write_results
from benchmarks.lib.data import fixed_l0_feature_acts
from jumprelu_sae_kernels import sparse_decode

# Same three configurations as bench_baselines, so the torch.compile comparison
# is point-for-point comparable with the cuSPARSE one. The first point is the
# base config (B=32, F=65536, D=768, L0=64) used by the headline results table.
POINTS = [
    dict(B=32, n_features=65536, d_model=768, L0=64),
    dict(B=256, n_features=65536, d_model=768, L0=64),
    dict(B=32, n_features=131072, d_model=512, L0=128),
]


def _dense(a, w):
    return a @ w


def main():
    env = capture_env()
    rows = []

    compiled_default = torch.compile(_dense)
    compiled_maxauto = torch.compile(_dense, mode="max-autotune-no-cudagraphs")

    for p in POINTS:
        B, F, D, L0 = p["B"], p["n_features"], p["d_model"], p["L0"]
        W = torch.randn(F, D, device="cuda").contiguous()
        acts = fixed_l0_feature_acts(B, F, L0)

        # Sanity: compiled dense must equal eager dense (same op).
        ref = acts @ W
        assert torch.allclose(compiled_default(acts, W), ref, atol=1e-3, rtol=1e-3)

        dense = bench(lambda: acts @ W)["median_ms"]
        comp = bench(lambda: compiled_default(acts, W))["median_ms"]
        comp_ma = bench(lambda: compiled_maxauto(acts, W))["median_ms"]
        ours_exact = bench(lambda: sparse_decode(acts, W, variant="exact"))["median_ms"]
        ours_fixed = bench(
            lambda: sparse_decode(acts, W, variant="fixed", max_l0=L0, validate=False)
        )["median_ms"]
        ours_fixed_val = bench(
            lambda: sparse_decode(acts, W, variant="fixed", max_l0=L0, validate=True)
        )["median_ms"]

        best_compile = min(comp, comp_ma)
        row = dict(
            **p,
            sparsity=L0 / F,
            dense_ms=dense,
            compile_default_ms=comp,
            compile_maxautotune_ms=comp_ma,
            compile_best_ms=best_compile,
            ours_exact_ms=ours_exact,
            ours_fixed_ms=ours_fixed,
            ours_fixed_validate_ms=ours_fixed_val,
            compile_best_speedup_vs_dense=dense / best_compile,
            exact_vs_best_compile=best_compile / ours_exact,
            fixed_vs_best_compile=best_compile / ours_fixed,
            fixed_validate_vs_best_compile=best_compile / ours_fixed_val,
        )
        rows.append(row)
        print(f"B{B} F{F} D{D} L{L0}:")
        print(f"  dense={dense:.3f} compile={comp:.3f} compile_ma={comp_ma:.3f} "
              f"(best={best_compile:.3f}, {dense / best_compile:.2f}x vs dense) "
              f"exact={ours_exact:.3f} fixed={ours_fixed:.3f} "
              f"fixed+val={ours_fixed_val:.3f}")
        print(f"  exact vs compile {row['exact_vs_best_compile']:.2f}x | "
              f"fixed vs compile {row['fixed_vs_best_compile']:.2f}x | "
              f"fixed+val vs compile {row['fixed_validate_vs_best_compile']:.2f}x")
    csv_path, _ = write_results(rows, "torch_compile", env)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
