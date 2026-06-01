import torch
import triton

from kernel_jumprelu_sae.fast.kernels import (
    compute_csr_kernel_fast,
    sparse_decode_kernel_fast,
)


def build_csr_fast(feature_acts: torch.Tensor, BLOCK_F: int = 1024, max_l0: int = 512):
    """Build a fixed-stride CSR layout from dense [B, n_features] activations.

    Each token gets a fixed region of `max_l0` slots at [b*max_l0, (b+1)*max_l0),
    so region starts are pure arithmetic (no count pass, no cumsum, no CPU sync).
    write_pos doubles as the per-token write cursor during the kernel and the
    per-token count afterward. Tokens firing more than max_l0 features have the
    overflow silently dropped (guarded, so no memory corruption) -- max_l0 must
    therefore be an upper bound on L0 for exact results.
    """
    B, n_features = feature_acts.shape
    device = feature_acts.device

    capacity = B * max_l0
    flat_idx = torch.empty(capacity, dtype=torch.int32, device=device)
    flat_val = torch.empty(capacity, dtype=feature_acts.dtype, device=device)

    write_pos = torch.zeros(B, dtype=torch.int32, device=device)

    grid = (B, triton.cdiv(n_features, BLOCK_F))
    compute_csr_kernel_fast[grid](
        feature_acts,
        write_pos,
        flat_idx,
        flat_val,
        n_features,
        max_l0,
        BLOCK_F=BLOCK_F,
    )

    counts = write_pos  # final cursor = per-token count (stays on GPU)
    return flat_idx, flat_val, counts, B, max_l0


def _sparse_decode_fast(flat_idx, flat_val, counts, W_dec, B, max_l0, BLOCK_D: int = 256):
    """Launch the batched CSR decoder. Internal: handles grid + output allocation."""
    d_model = W_dec.shape[1]
    out = torch.zeros((B, d_model), device=W_dec.device, dtype=torch.float32)

    grid = (B, triton.cdiv(d_model, BLOCK_D))
    sparse_decode_kernel_fast[grid](
        flat_idx, flat_val, counts, W_dec, out, d_model, max_l0, BLOCK_D=BLOCK_D
    )
    return out


def sparse_decode_fast(feature_acts, W_dec, max_l0: int = 512):
    """Sparse replacement for `feature_acts @ W_dec`.

    Args:
        feature_acts: [B, n_features] sparse activations.
        W_dec: [n_features, d_model] decoder weights.
        max_l0: fixed per-token capacity; must be >= the max L0 across tokens
            for exact results (overflow is silently dropped otherwise).
    Returns:
        [B, d_model] reconstruction, equal to feature_acts @ W_dec (when no overflow).
    """
    W_dec = W_dec.contiguous()
    flat_idx, flat_val, counts, B, ml0 = build_csr_fast(feature_acts, max_l0=max_l0)
    return _sparse_decode_fast(flat_idx, flat_val, counts, W_dec, B, ml0)
