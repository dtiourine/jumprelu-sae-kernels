"""Shared fixtures and helpers for the kernel test suite.

Provides: the CUDA skip gate, a seeded/deterministic sparse-activation
generator (supporting ragged per-token L0), an fp32 correctness oracle, an
fp64 accuracy oracle, per-dtype tolerances, and a variant-dispatch helper that
routes every test through the public api.sparse_decode entry point.
"""

import pytest
import torch

from jumprelu_sae_kernels import sparse_decode

DEVICE = "cuda"
VARIANTS = ["exact", "fixed"]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels require a CUDA GPU"
)


def make_sparse_acts(B, n_features, l0_per_token, *, dtype=torch.float32, seed=0):
    """[B, n_features] sparse activations.

    l0_per_token: int (same L0 for every token) or list[int] of length B
    (ragged per-token counts). Deterministic given seed.
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    acts = torch.zeros(B, n_features, device=DEVICE, dtype=dtype)
    if isinstance(l0_per_token, int):
        l0_per_token = [l0_per_token] * B
    assert len(l0_per_token) == B
    for b, L0 in enumerate(l0_per_token):
        if L0 > 0:
            fired = torch.randperm(n_features, generator=g, device=DEVICE)[:L0]
            acts[b, fired] = torch.randn(L0, generator=g, device=DEVICE, dtype=dtype)
    return acts


def dense_fp32_ref(feature_acts, W_dec):
    """Correctness oracle: fp32 dense matmul, matching the kernel's fp32 accum."""
    return feature_acts.float() @ W_dec.float()


def fp64_ref(feature_acts, W_dec):
    """Accuracy oracle: fp64 ground truth."""
    return feature_acts.double() @ W_dec.double()


def dtype_tol(dtype):
    """assert_close tolerances vs the fp32 reference. The kernel upcasts
    operands to fp32 before the multiply and accumulates in fp32, so its
    accuracy is independent of the input dtype — fp16/bf16 inputs get the same
    fp32-quality result as fp32. Tolerances are therefore uniform (fp32-level);
    they only absorb fp32 accumulation-order differences vs the reference at the
    L0 values used in the correctness suite (<=100). (dtype is accepted for
    call-site symmetry but no longer affects the tolerance.)"""
    return dict(atol=1e-4, rtol=1e-3)


def decode_variant(feature_acts, W_dec, variant, max_l0=None):
    """Route through the public API. For 'fixed', default max_l0 to the actual
    max L0 in this batch so the correct-result path is exercised (tests that
    target the overflow path pass an explicit smaller max_l0)."""
    if variant == "fixed":
        if max_l0 is None:
            max_l0 = int((feature_acts != 0).sum(dim=1).max().item())
            max_l0 = max(max_l0, 1)
        return sparse_decode(feature_acts, W_dec, variant="fixed", max_l0=max_l0)
    return sparse_decode(feature_acts, W_dec, variant="exact")
