"""Layer 0 — public-contract correctness, parametrized over variant x dtype,
routed through api.sparse_decode. Plus cross-variant equivalence and
determinism."""

import pytest
import torch

from conftest import (
    DEVICE,
    VARIANTS,
    DTYPES,
    requires_cuda,
    make_sparse_acts,
    dense_fp32_ref,
    dtype_tol,
    decode_variant,
)

pytestmark = requires_cuda


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("B", [1, 4, 32])
@pytest.mark.parametrize("n_features", [256, 1024, 16384])
@pytest.mark.parametrize("d_model", [128, 512, 768])
@pytest.mark.parametrize("L0", [1, 8, 100])
def test_matches_dense(variant, dtype, B, n_features, d_model, L0):
    W_dec = torch.randn(n_features, d_model, device=DEVICE, dtype=dtype)
    acts = make_sparse_acts(B, n_features, L0, dtype=dtype)
    out = decode_variant(acts, W_dec, variant, max_l0=L0)
    ref = dense_fp32_ref(acts, W_dec)
    assert out.shape == ref.shape
    assert out.dtype == torch.float32  # output invariant
    torch.testing.assert_close(out, ref, **dtype_tol(dtype))


@pytest.mark.parametrize("variant", VARIANTS)
def test_ragged_batch(variant):
    """Different L0 per token, including a zero-firing token — the case the
    ragged/CSR layout exists for."""
    B, n_features, d_model = 5, 1024, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    l0 = [0, 1, 50, 200, 17]
    acts = make_sparse_acts(B, n_features, l0)
    out = decode_variant(acts, W_dec, variant, max_l0=max(l0))
    torch.testing.assert_close(out, dense_fp32_ref(acts, W_dec), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_features_fired(variant):
    B, n_features, d_model = 4, 512, 256
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, n_features)
    out = decode_variant(acts, W_dec, variant, max_l0=n_features)
    torch.testing.assert_close(out, dense_fp32_ref(acts, W_dec), atol=1e-3, rtol=1e-3)


def test_variant_equivalence():
    """exact and fixed compute the same thing for L0 <= max_l0. They sum the
    same feature set in fp32; summation order may differ between variants, so
    we assert agreement to fp32 round-off. If they turn out bit-identical in
    practice on this hardware, tighten this to torch.equal."""
    B, n_features, d_model = 8, 4096, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, l0_per_token=[10, 0, 64, 5, 100, 1, 33, 200])
    out_exact = decode_variant(acts, W_dec, "exact")
    out_fixed = decode_variant(acts, W_dec, "fixed", max_l0=256)
    torch.testing.assert_close(out_exact, out_fixed, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", VARIANTS)
def test_determinism(variant):
    """Repeated runs of the same variant are bit-identical despite atomic
    cursors in CSR build: decode is a per-token fp32 reduction over the same
    feature set each run."""
    B, n_features, d_model, L0 = 4, 1024, 512, 50
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, L0)
    out1 = decode_variant(acts, W_dec, variant, max_l0=L0)
    out2 = decode_variant(acts, W_dec, variant, max_l0=L0)
    assert torch.equal(out1, out2)
