import torch

from benchmarks.lib.harness import capture_env, bench, write_results
from benchmarks.lib.data import fixed_l0_feature_acts
from jumprelu_sae_kernels import sparse_decode

BASE = dict(B=32, n_features=65536, d_model=768)
L0S = [16, 64, 256, 1024, 4096]


def _dense(a, w):
    return a @ w


def main():
    env = capture_env()
    rows = []

    compiled_default = torch.compile(_dense)
    compiled_maxauto = torch.compile(_dense, mode="max-autotune-no-cudagraphs")

    for L0 in L0S:
        B, F, D = BASE["B"], BASE["n_features"], BASE["d_model"]
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

        best_compile = min(comp, comp_ma)
        row = dict(
            B=B, n_features=F, d_model=D, L0=L0,
            sparsity=L0 / F,
            dense_ms=dense,
            compile_default_ms=comp,
            compile_maxautotune_ms=comp_ma,
            ours_exact_ms=ours_exact,
            ours_fixed_ms=ours_fixed,
            fixed_vs_dense=dense / ours_fixed,
            fixed_vs_best_compile=best_compile / ours_fixed,
            exact_vs_best_compile=best_compile / ours_exact,
            compile_best_vs_dense=dense / best_compile,
        )
        rows.append(row)
        print(
            f"L0={L0}: dense={dense:.3f} compile={comp:.3f} "
            f"compile_ma={comp_ma:.3f} exact={ours_exact:.3f} fixed={ours_fixed:.3f} "
            f"(fixed vs dense {row['fixed_vs_dense']:.2f}x | "
            f"fixed vs best-compile {row['fixed_vs_best_compile']:.2f}x | "
            f"compile vs dense {row['compile_best_vs_dense']:.2f}x)"
        )
    csv_path, _ = write_results(rows, "torch_compile", env)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
