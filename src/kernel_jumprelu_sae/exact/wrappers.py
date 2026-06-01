import triton
import torch

from kernel_jumprelu_sae.exact.kernels import (
    compute_csr_kernel,
    count_nonzero,
    sparse_decode_kernel,
)


def build_csr(feature_acts: torch.Tensor, BLOCK_F: int = 1024):
    B, n_features = feature_acts.shape
    device = feature_acts.device

    counts = torch.zeros(B, dtype=torch.int32, device=device)
    grid = (B, triton.cdiv(n_features, BLOCK_F))
    count_nonzero[grid](feature_acts, counts, n_features, BLOCK_F=BLOCK_F)

    row_offsets = torch.zeros(B + 1, dtype=torch.int32, device=device)
    row_offsets[1:] = counts.cumsum(0).to(torch.int32)

    # The last entry of row_offsets is the total number of nonzeros across the batch
    total_nnz = int(row_offsets[-1].item())

    flat_idx = torch.empty(total_nnz, dtype=torch.int32, device=device)
    flat_val = torch.empty(total_nnz, dtype=feature_acts.dtype, device=device)

    # write_pos is a per-token shared cursor: a token's blocks run concurrently
    # and all append into that token's region, so they use this to coordinate the
    # next free slot. Each block atomically reads the current value (its write
    # offset within the token) and bumps it by how many features it wrote.
    # Starts at 0 since blocks only ever add to it.
    write_pos = torch.zeros(B, dtype=torch.int32, device=device)

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
    flat_idx, flat_val, row_offsets, B = build_csr(feature_acts)
    return _sparse_decode(flat_idx, flat_val, row_offsets, W_dec, B)
