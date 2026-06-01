"""Layer 2 — edge cases and the inference-only contract."""

import pytest
import torch

from conftest import (
    DEVICE,
    VARIANTS,
    requires_cuda,
    make_sparse_acts,
    dense_fp32_ref,
    decode_variant,
)
from jumprelu_sae_kernels import sparse_decode

pytestmark = requires_cuda


@pytest.mark.parametrize("variant", VARIANTS)
def test_requires_grad_raises(variant):
    """Inference-only contract: grad-requiring inputs must raise, never
    silently detach (which is what the code did before this guard)."""
    n_features, d_model = 256, 64
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(4, n_features, 8)
    acts.requires_grad_(True)
    with pytest.raises(RuntimeError, match="inference-only"):
        decode_variant(acts, W_dec, variant, max_l0=32)


@pytest.mark.parametrize("variant", VARIANTS)
def test_requires_grad_on_w_dec_raises(variant):
    n_features, d_model = 256, 64
    W_dec = torch.randn(n_features, d_model, device=DEVICE, requires_grad=True)
    acts = make_sparse_acts(4, n_features, 8)
    with pytest.raises(RuntimeError, match="inference-only"):
        decode_variant(acts, W_dec, variant, max_l0=32)


@pytest.mark.parametrize("variant", VARIANTS)
def test_no_grad_inference_with_grad_params_ok(variant):
    """The guard is grad-context-aware: under torch.no_grad(), inference runs
    normally even when params carry requires_grad=True (the common case — e.g.
    a stock SAELens SAE whose W_dec is an nn.Parameter). No detach required."""
    n_features, d_model = 256, 64
    W_dec = torch.randn(n_features, d_model, device=DEVICE, requires_grad=True)
    acts = make_sparse_acts(4, n_features, 8)
    with torch.no_grad():
        out = decode_variant(acts, W_dec, variant, max_l0=32)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, dense_fp32_ref(acts, W_dec), atol=1e-4, rtol=1e-3)


def test_fixed_overflow_raises():
    """fixed variant must raise when a token fires more than max_l0 (would
    otherwise truncate silently)."""
    B, n_features, d_model = 3, 1024, 256
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, l0_per_token=[5, 50, 9])
    with pytest.raises(ValueError, match="max_l0"):
        sparse_decode(acts, W_dec, variant="fixed", max_l0=49)


def test_fixed_overflow_boundary_exact_fit():
    """L0 == max_l0 is allowed and correct."""
    B, n_features, d_model = 3, 1024, 256
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, l0_per_token=[10, 50, 30])
    out = sparse_decode(acts, W_dec, variant="fixed", max_l0=50)
    torch.testing.assert_close(out, dense_fp32_ref(acts, W_dec), atol=1e-4, rtol=1e-3)


def test_fixed_overflow_does_not_corrupt_neighbor():
    """A token overflowing must not be caught only by luck of memory layout;
    when max_l0 bounds all tokens, every token (including those adjacent to a
    near-capacity one) is correct."""
    B, n_features, d_model = 3, 1024, 256
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, l0_per_token=[100, 1, 100])
    out = sparse_decode(acts, W_dec, variant="fixed", max_l0=100)
    torch.testing.assert_close(
        out[1], dense_fp32_ref(acts, W_dec)[1], atol=1e-4, rtol=1e-3
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_non_contiguous_w_dec(variant):
    """Wrappers call .contiguous(); a transposed view must still be correct."""
    n_features, d_model = 1024, 512
    base = torch.randn(d_model, n_features, device=DEVICE)
    W_dec = base.t()  # non-contiguous [n_features, d_model] view
    assert not W_dec.is_contiguous()
    acts = make_sparse_acts(4, n_features, 50)
    out = decode_variant(acts, W_dec, variant, max_l0=50)
    torch.testing.assert_close(out, dense_fp32_ref(acts, W_dec), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_entire_batch_empty(variant):
    """No token fires -> all zeros, no crash."""
    B, n_features, d_model = 4, 1024, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = torch.zeros(B, n_features, device=DEVICE)
    out = decode_variant(acts, W_dec, variant, max_l0=1)
    torch.testing.assert_close(out, torch.zeros(B, d_model, device=DEVICE))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("d_model", [1, 255, 257, 500, 513])
def test_d_model_tail_masking(variant, d_model):
    """d_model not a multiple of BLOCK_D exercises tail masking."""
    B, n_features, L0 = 4, 1024, 50
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, L0)
    out = decode_variant(acts, W_dec, variant, max_l0=L0)
    torch.testing.assert_close(out, dense_fp32_ref(acts, W_dec), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("variant", VARIANTS)
def test_b1_single_token(variant):
    n_features, d_model = 1024, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(1, n_features, 50)
    out = decode_variant(acts, W_dec, variant, max_l0=50)
    torch.testing.assert_close(out, dense_fp32_ref(acts, W_dec), atol=1e-4, rtol=1e-3)
