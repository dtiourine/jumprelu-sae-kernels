"""Layer 1 — accuracy characterization against an fp64 reference.

The kernel accumulates in fp32. We characterize its reconstruction accuracy with
the relative Frobenius-norm error  ||out - ref|| / ||ref||  against an fp64
reference. This metric is stable and output-scale-aware, unlike max relative
error (which is dominated by whichever output element happens to be nearest
zero, and swings wildly across random seeds).

Enforced claim (per dtype): the kernel's relative Frobenius error stays below an
absolute threshold comfortably within that dtype's round-off. That is the real
correctness guarantee. We additionally REPORT the kernel-vs-cuBLAS ratio for the
writeup: the kernel is within a small constant factor of cuBLAS (cuBLAS uses
blocked/pairwise accumulation; the kernel sums sequentially in fp32), and both
errors are negligible for SAE reconstruction. We deliberately do NOT assert a
tight bound on that ratio — it reflects cuBLAS's internals, not a kernel defect.

Run as a report:  uv run pytest tests/test_accuracy.py -s -k report
"""

import pytest
import torch

from conftest import (
    DEVICE,
    VARIANTS,
    DTYPES,
    requires_cuda,
    make_sparse_acts,
    fp64_ref,
    decode_variant,
)

pytestmark = requires_cuda

# Ceiling on the kernel's relative Frobenius error vs fp64. The kernel upcasts
# operands to fp32 before multiplying and accumulates in fp32, so error is
# dtype-independent (the fp64 reference uses the same dtype-rounded inputs, so
# only the fp32 arithmetic round-off remains). Worst observed across SHAPES x
# variants x dtypes at SEED is ~1.1e-6; the bound is set a few x above that —
# tight enough that any real correctness bug (relF ~ O(0.1-1)) trips it.
REL_F_MAX = {
    torch.float32: 1e-5,
    torch.float16: 1e-5,
    torch.bfloat16: 1e-5,
}

SEED = 1234

SHAPES = [
    # (B, n_features, d_model, L0)
    (32, 16384, 768, 64),
    (32, 16384, 768, 512),
    (8, 65536, 512, 128),
    (4, 4096, 256, 4096),  # dense extreme
]


def _make(B, n_features, d_model, L0, dtype):
    """Seeded acts + W_dec for a deterministic accuracy measurement."""
    acts = make_sparse_acts(B, n_features, L0, dtype=dtype, seed=SEED)
    g = torch.Generator(device=DEVICE).manual_seed(SEED + 1)
    W_dec = torch.randn(n_features, d_model, device=DEVICE, dtype=dtype, generator=g)
    return acts, W_dec


def _rel_frobenius(out, ref):
    """||out - ref|| / ||ref||, computed in fp64. ref is already fp64."""
    out = out.double()
    denom = ref.norm().item()
    return (out - ref).norm().item() / max(denom, 1e-30)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"B{s[0]}_F{s[1]}_D{s[2]}_L{s[3]}")
def test_kernel_accuracy_within_dtype_bound(variant, dtype, shape):
    """The real correctness guarantee: kernel rel-Frobenius error is within the
    dtype's round-off bound."""
    B, n_features, d_model, L0 = shape
    acts, W_dec = _make(B, n_features, d_model, L0, dtype)
    ref = fp64_ref(acts, W_dec)
    out_kernel = decode_variant(acts, W_dec, variant, max_l0=L0)
    relF = _rel_frobenius(out_kernel, ref)
    bound = REL_F_MAX[dtype]
    assert relF <= bound, (
        f"{variant}/{dtype}/{shape}: kernel rel-Frobenius {relF:.2e} > bound {bound:.2e}"
    )


def test_report(capsys):
    """Prints the kernel-vs-cuBLAS rel-Frobenius table (vs fp64) for the writeup.
    Always passes; run with -s to see it."""
    rows = []
    for variant in VARIANTS:
        for dtype in DTYPES:
            for (B, n_features, d_model, L0) in SHAPES:
                acts, W_dec = _make(B, n_features, d_model, L0, dtype)
                ref = fp64_ref(acts, W_dec)
                relF_k = _rel_frobenius(decode_variant(acts, W_dec, variant, max_l0=L0), ref)
                relF_d = _rel_frobenius(acts @ W_dec, ref)
                ratio = relF_k / max(relF_d, 1e-30)
                rows.append(
                    f"{variant:5s} {str(dtype).replace('torch.', ''):9s} "
                    f"B{B} F{n_features} D{d_model} L{L0:5d} | "
                    f"relF_kernel={relF_k:.2e} relF_dense={relF_d:.2e} ratio={ratio:5.2f}"
                )
    with capsys.disabled():
        print("\n--- accuracy report: rel-Frobenius error vs fp64 (kernel vs cuBLAS) ---")
        print("\n".join(rows))
