from kernel_jumprelu_sae.kernel import sparse_decode_kernel
import triton
import torch


def _sparse_decode(flat_idx, flat_val, row_offsets, W_dec, B, BLOCK_D=256):
    d_model = W_dec.shape[1]
    out = torch.zeros((B, d_model), device=W_dec.device, dtype=torch.float32)

    grid = (
        B,
        triton.cdiv(d_model, BLOCK_D),
    )

    sparse_decode_kernel[grid](
        flat_idx, flat_val, row_offsets, W_dec, out, d_model, BLOCK_D=BLOCK_D
    )

    return out


def sparse_decode(feature_acts, W_dec):
    B, n_features = feature_acts.shape

    W_dec = W_dec.contiguous()

    # nonzero on [B, n_features] -> [total_nnz, 2]: (token_id, feature_id), token-sorted
    coords = feature_acts.nonzero()
    token_ids = coords[:, 0]
    feat_ids = coords[:, 1]
    flat_idx = feat_ids.to(torch.int32)
    flat_val = feature_acts[token_ids, feat_ids]

    # CSR offsets: per-token counts -> cumulative sum, length B+
    counts = torch.bincount(token_ids, minlength=B)
    row_offsets = torch.zeros(B + 1, dtype=torch.int32, device=feature_acts.device)
    row_offsets[1:] = counts.cumsum(0).to(torch.int32)

    flat_idx = flat_idx.contiguous()
    flat_val = flat_val.contiguous()

    return _sparse_decode(flat_idx, flat_val, row_offsets, W_dec, B)
