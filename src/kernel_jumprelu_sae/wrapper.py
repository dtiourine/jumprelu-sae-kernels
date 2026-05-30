from kernel_jumprelu_sae.kernels.sparse_decode import sparse_decode_kernel
import triton
import torch

from kernel_jumprelu_sae.kernels.compute_csr import (
    compute_csr_kernel,
    count_nonsparse_elements,
)


def build_csr_fused(feature_acts, BLOCK_F=1024):
    """Build CSR arrays (flat_idx, flat_val, row_offsets, B) from dense
    [B, n_features] activations using the two-pass fused kernels.

    Pass 1 counts nonzeros per token; a cumsum turns counts into row_offsets;
    Pass 2 scatters each token's fired features into its reserved region.
    """
    B, n_features = feature_acts.shape
    device = feature_acts.device

    # --- pass 1: count nonzeros per token ---
    counts = torch.zeros(
        B, dtype=torch.int32, device=device
    )  # atomics accumulate -> must be 0
    grid = (B, triton.cdiv(n_features, BLOCK_F))
    count_nonsparse_elements[grid](feature_acts, counts, n_features, BLOCK_F=BLOCK_F)

    # --- counts -> row_offsets (length B+1) ---
    row_offsets = torch.zeros(B + 1, dtype=torch.int32, device=device)
    row_offsets[1:] = counts.cumsum(0).to(torch.int32)

    # --- allocate the flat output arrays, sized by total nonzeros ---
    total_nnz = int(row_offsets[-1].item())  # syncs once; see note below
    flat_idx = torch.empty(total_nnz, dtype=torch.int32, device=device)
    flat_val = torch.empty(total_nnz, dtype=feature_acts.dtype, device=device)

    # --- pass 2: scatter into reserved regions ---
    write_pos = torch.zeros(
        B, dtype=torch.int32, device=device
    )  # per-token write cursor, init 0
    compute_csr_kernel[grid](
        feature_acts,
        row_offsets,
        write_pos,
        flat_idx,
        flat_val,
        n_features,
        BLOCK_F=BLOCK_F,
    )

    return flat_idx, flat_val, row_offsets, B


def _sparse_decode(flat_idx, flat_val, row_offsets, W_dec, B, BLOCK_D=256):
    """Launch the batched CSR kernel. Internal: handles grid + output allocation."""
    d_model = W_dec.shape[1]
    out = torch.zeros((B, d_model), device=W_dec.device, dtype=torch.float32)

    grid = (B, triton.cdiv(d_model, BLOCK_D))

    sparse_decode_kernel[grid](
        flat_idx, flat_val, row_offsets, W_dec, out, d_model, BLOCK_D=BLOCK_D
    )

    return out


def sparse_decode(feature_acts, W_dec):
    """Sparse replacement for `feature_acts @ W_dec`.

    Args:
        feature_acts: [B, n_features] sparse activations.
        W_dec: [n_features, d_model] decoder weights.

    Returns:
        [B, d_model] reconstruction, equal to feature_acts @ W_dec.
    """
    W_dec = W_dec.contiguous()
    flat_idx, flat_val, row_offsets, B = build_csr_fused(feature_acts)
    return _sparse_decode(flat_idx, flat_val, row_offsets, W_dec, B)
