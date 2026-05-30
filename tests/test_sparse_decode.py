"""Correctness tests for the batched sparse JumpReLU SAE decoder.

Public contract:  sparse_decode(feature_acts, W_dec) == feature_acts @ W_dec
where feature_acts is [B, n_features] (sparse) and the result is [B, d_model].
Verified against the dense matmul reference up to float tolerance.

Run:  pytest tests/test_sparse_decode.py -v
"""

import pytest
import torch

from kernel_jumprelu_sae.wrapper import sparse_decode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires a CUDA GPU"
)
DEVICE = "cuda"


def make_sparse_acts(
    B, n_features, l0_per_token, device=DEVICE, dtype=torch.float32, seed=0
):
    """Build [B, n_features] sparse activations.

    l0_per_token: int (same count for every token) or list[int] of length B
    (per-token counts, for testing ragged batches).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    acts = torch.zeros(B, n_features, device=device, dtype=dtype)
    if isinstance(l0_per_token, int):
        l0_per_token = [l0_per_token] * B
    for b, L0 in enumerate(l0_per_token):
        if L0 > 0:
            fired = torch.randperm(n_features, generator=g, device=device)[:L0]
            acts[b, fired] = torch.randn(L0, generator=g, device=device, dtype=dtype)
    return acts


def reference(feature_acts, W_dec):
    return feature_acts @ W_dec


# --- core correctness sweep ------------------------------------------------
@pytest.mark.parametrize("B", [1, 4, 32])
@pytest.mark.parametrize("n_features", [256, 1024, 16384])
@pytest.mark.parametrize("d_model", [128, 512, 768])
@pytest.mark.parametrize("L0", [1, 8, 100])
def test_matches_dense(B, n_features, d_model, L0):
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, L0)
    out = sparse_decode(acts, W_dec)
    ref = reference(acts, W_dec)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)


# --- the test that actually exercises the CSR / ragged layout --------------
def test_ragged_batch():
    """Different L0 per token in one batch, including a zero-firing token.
    This is the case the CSR row_offsets exist to handle."""
    B, n_features, d_model = 5, 1024, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, l0_per_token=[0, 1, 50, 200, 17])
    out = sparse_decode(acts, W_dec)
    ref = reference(acts, W_dec)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)


def test_b1_matches_single_token():
    """B=1 should reduce to the single-token result."""
    n_features, d_model = 1024, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(1, n_features, 50)
    out = sparse_decode(acts, W_dec)
    ref = reference(acts, W_dec)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)


# --- edge cases ------------------------------------------------------------
def test_entire_batch_empty():
    """No token fires anything -> all zeros, no crash."""
    B, n_features, d_model = 4, 1024, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = torch.zeros(B, n_features, device=DEVICE)
    out = sparse_decode(acts, W_dec)
    torch.testing.assert_close(out, torch.zeros(B, d_model, device=DEVICE))


@pytest.mark.parametrize("d_model", [1, 255, 257, 500, 513])
def test_d_model_not_divisible_by_block(d_model):
    """Tail-masking when d_model isn't a multiple of BLOCK_D."""
    B, n_features, L0 = 4, 1024, 50
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, L0)
    out = sparse_decode(acts, W_dec)
    ref = reference(acts, W_dec)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)


def test_all_features_fired():
    B, n_features, d_model = 4, 512, 256
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, l0_per_token=n_features)
    out = sparse_decode(acts, W_dec)
    ref = reference(acts, W_dec)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)


def test_determinism():
    B, n_features, d_model, L0 = 4, 1024, 512, 50
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    acts = make_sparse_acts(B, n_features, L0)
    out1 = sparse_decode(acts, W_dec)
    out2 = sparse_decode(acts, W_dec)
    assert torch.equal(out1, out2)
