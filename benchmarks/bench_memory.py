import torch

from benchmarks.lib.harness import capture_env, peak_memory, write_results
from benchmarks.lib.data import fixed_l0_feature_acts
from jumprelu_sae_kernels import sparse_decode

N_FEATURES, D_MODEL, L0 = 65536, 768, 64
BATCHES = [32, 256, 1024]
MAX_L0S = [64, 256, 512, 1024]


def main():
    env = capture_env()
    rows = []
    for B in BATCHES:
        W = torch.randn(N_FEATURES, D_MODEL, device="cuda").contiguous()
        acts = fixed_l0_feature_acts(B, N_FEATURES, L0)
        dense_mem = peak_memory(lambda: acts @ W)
        exact_mem = peak_memory(lambda: sparse_decode(acts, W, variant="exact"))
        for ml in MAX_L0S:
            fixed_mem = peak_memory(
                lambda: sparse_decode(acts, W, variant="fixed", max_l0=ml, validate=False)
            )
            rows.append(
                dict(
                    B=B,
                    n_features=N_FEATURES,
                    d_model=D_MODEL,
                    L0=L0,
                    max_l0=ml,
                    dense_bytes=dense_mem,
                    exact_bytes=exact_mem,
                    fixed_bytes=fixed_mem,
                    fixed_over_exact=fixed_mem / max(exact_mem, 1),
                )
            )
            print(
                f"B={B} max_l0={ml}: dense={dense_mem/1e6:.1f}MB "
                f"exact={exact_mem/1e6:.1f}MB fixed={fixed_mem/1e6:.1f}MB"
            )
    csv_path, _ = write_results(rows, "memory", env)
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
