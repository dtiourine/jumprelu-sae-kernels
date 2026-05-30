from kernel_jumprelu_sae.kernel import sparse_decode_kernel
import triton
import torch


def build_csr(feature_acts):
    """Convert dense activations into the CSR layout the kernel consumes.

    Args:
        feature_acts: [B, n_features] sparse activations (mostly zero).

    Returns:
        flat_idx: int32 [total_nnz], all tokens' fired-feature indices,
            concatenated in token order.
        flat_val: float [total_nnz], their values, aligned with flat_idx.
        row_offsets: int32 [B + 1], CSR offsets; token b's features are the
            slice [row_offsets[b], row_offsets[b + 1]) of the flat arrays.
        B: int, the batch size.
    """
    B, n_features = feature_acts.shape

    # nonzero on [B, n_features] -> [total_nnz, 2]: (token_id, feature_id), token-sorted
    coords = feature_acts.nonzero()
    token_ids = coords[:, 0]
    feat_ids = coords[:, 1]
    flat_idx = feat_ids.to(torch.int32).contiguous()
    flat_val = feature_acts[token_ids, feat_ids].contiguous()

    # CSR offsets: per-token counts -> cumulative sum, length B + 1
    counts = torch.bincount(token_ids, minlength=B)
    row_offsets = torch.zeros(B + 1, dtype=torch.int32, device=feature_acts.device)
    row_offsets[1:] = counts.cumsum(0).to(torch.int32)

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
    flat_idx, flat_val, row_offsets, B = build_csr(feature_acts)
    return _sparse_decode(flat_idx, flat_val, row_offsets, W_dec, B)
