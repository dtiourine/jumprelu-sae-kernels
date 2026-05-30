"""Correctness tests for the sparse JumpReLU SAE decoder kernel.

Reference: dense  x_hat = h @ W_dec  (+ b_dec)
The kernel must match this for any sparse h, up to floating-point tolerance.

Run:  pytest tests/test_sparse_decode.py -v
"""

import pytest
import torch

from kernel_jumprelu_sae.wrapper import sparse_decode

# ---- skip the whole module if no GPU (the kernel can't run on CPU) ----
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernel requires a CUDA GPU"
)

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_sparse_h(n_features, L0, device=DEVICE, dtype=torch.float32, seed=0):
    """Build a sparse activation vector with exactly L0 active features,
    and return (h_dense, idx, val) where idx/val are the compact form."""
    g = torch.Generator(device=device).manual_seed(seed)
    h = torch.zeros(n_features, device=device, dtype=dtype)
    if L0 > 0:
        fired = torch.randperm(n_features, generator=g, device=device)[:L0]
        h[fired] = torch.randn(L0, generator=g, device=device, dtype=dtype)
    idx = h.nonzero().squeeze(-1).to(torch.int32)
    val = h[idx]
    return h, idx, val


def reference_decode(h, W_dec):
    """Trusted ground-truth: plain dense matmul."""
    return h @ W_dec


# ---------------------------------------------------------------------------
# core correctness across a sweep of shapes and sparsity levels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_features", [256, 1024, 16384])
@pytest.mark.parametrize("d_model", [128, 512, 768])  # incl. non-power-of-2
@pytest.mark.parametrize("L0", [1, 8, 100])
def test_matches_dense_reference(n_features, d_model, L0):
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    h, idx, val = make_sparse_h(n_features, L0)

    out = sparse_decode(idx, val, W_dec)
    ref = reference_decode(h, W_dec)

    assert out.shape == ref.shape, f"shape {out.shape} != {ref.shape}"
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# edge cases — the silent-bug hunters
# ---------------------------------------------------------------------------
def test_no_features_fired():
    """L0 = 0: output must be all zeros, not garbage."""
    n_features, d_model = 1024, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    h, idx, val = make_sparse_h(n_features, L0=0)

    out = sparse_decode(idx, val, W_dec)
    assert torch.allclose(out, torch.zeros_like(out)), "empty input should give zeros"


def test_single_feature():
    """L0 = 1: output should equal val * that single row."""
    n_features, d_model = 1024, 512
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    h, idx, val = make_sparse_h(n_features, L0=1)

    out = sparse_decode(idx, val, W_dec)
    expected = val[0] * W_dec[idx[0].long()]
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("d_model", [1, 255, 257, 500, 513])
def test_d_model_not_divisible_by_block(d_model):
    """Tail-masking: d_model not a multiple of BLOCK_D must still be correct."""
    n_features, L0 = 1024, 50
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    h, idx, val = make_sparse_h(n_features, L0)

    out = sparse_decode(idx, val, W_dec)
    ref = reference_decode(h, W_dec)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)


def test_all_features_fired():
    """L0 = n_features: the fully dense case, kernel must still match."""
    n_features, d_model = 512, 256
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    h, idx, val = make_sparse_h(n_features, L0=n_features)

    out = sparse_decode(idx, val, W_dec)
    ref = reference_decode(h, W_dec)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)  # more terms → looser


@pytest.mark.parametrize("BLOCK_D", [32, 64, 128, 256, 1024])
def test_various_block_sizes(BLOCK_D):
    """Result must be independent of the tile width."""
    n_features, d_model, L0 = 1024, 512, 50
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    h, idx, val = make_sparse_h(n_features, L0)

    out = sparse_decode(idx, val, W_dec, BLOCK_D=BLOCK_D)
    ref = reference_decode(h, W_dec)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3)


def test_determinism():
    """Same inputs → identical outputs across runs."""
    n_features, d_model, L0 = 1024, 512, 50
    W_dec = torch.randn(n_features, d_model, device=DEVICE)
    h, idx, val = make_sparse_h(n_features, L0)

    out1 = sparse_decode(idx, val, W_dec)
    out2 = sparse_decode(idx, val, W_dec)
    assert torch.equal(out1, out2), "kernel should be deterministic"
