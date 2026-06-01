import triton
import triton.language as tl


@triton.jit
def count_nonzero(feature_acts_ptr, counts_ptr, n_features, BLOCK_F: tl.constexpr):
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)
    feat_offsets = pid_d * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = feat_offsets < n_features

    feat_ptrs = feature_acts_ptr + pid_token * n_features + feat_offsets
    vals = tl.load(feat_ptrs, mask=mask, other=0.0)
    fired = vals != 0.0
    fired_count = tl.sum(fired.to(tl.int32))
    tl.atomic_add(counts_ptr + pid_token, fired_count)


@triton.jit
def compute_csr_kernel(
    feature_acts_ptr,
    row_offsets_ptr,
    write_pos_ptr,
    flat_idx_ptr,
    flat_val_ptr,
    n_features,
    BLOCK_F: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)
    feat_offsets = pid_d * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = feat_offsets < n_features

    feat_ptrs = feature_acts_ptr + pid_token * n_features + feat_offsets
    vals = tl.load(feat_ptrs, mask=mask, other=0.0)
    fired = vals != 0.0
    fired_int = fired.to(tl.int32)

    region_start = tl.load(row_offsets_ptr + pid_token)
    block_count = tl.sum(fired_int)

    base = tl.atomic_add(write_pos_ptr + pid_token, block_count)

    local_rank = tl.cumsum(fired_int) - fired_int

    slots = region_start + base + local_rank

    tl.store(flat_idx_ptr + slots, feat_offsets.to(tl.int32), mask=fired & mask)
    tl.store(flat_val_ptr + slots, vals, mask=fired & mask)


@triton.jit
def sparse_decode_kernel(
    flat_idx_ptr,
    flat_val_ptr,
    row_offsets_ptr,
    W_dec_ptr,
    out_ptr,
    d_model,
    BLOCK_D: tl.constexpr,
):
    """Batched sparse SAE decoder (CSR layout).

    For each token b, computes the reconstruction as a weighted sum of the
    decoder rows belonging to the features that fired:

        out[b, :] = sum_{j in token b's chunk}  flat_val[j] * W_dec[flat_idx[j], :]

    Tokens may fire different numbers of features (ragged); the CSR
    row_offsets array marks where each token's chunk lives in the flat
    arrays. The grid is 2D: axis 0 = token, axis 1 = output-column block.
    Each program owns one (token, BLOCK_D-wide output slice) pair and loops
    over that token's fired features, accumulating each contribution.

    Args:
        flat_idx_ptr: Concatenated fired-feature indices for all tokens.
        flat_val_ptr: Their values, aligned element-wise with flat_idx_ptr.
        row_offsets_ptr: CSR offsets of length B+1; token b's features are
            the slice [row_offsets[b], row_offsets[b+1]) of the flat arrays.
        W_dec_ptr: Decoder weight matrix [n_features, d_model], row-major.
        out_ptr: Output buffer [B, d_model], written in place.
        d_model: Output dimension (row width of W_dec).
        BLOCK_D: Compile-time tile width (output columns per program).
    """
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)

    start = tl.load(row_offsets_ptr + pid_token)
    end = tl.load(row_offsets_ptr + pid_token + 1)
    n = end - start

    offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < d_model
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in range(n):
        j = start + i
        feat_idx = tl.load(flat_idx_ptr + j)
        feat_val = tl.load(flat_val_ptr + j)
        row_ptrs = W_dec_ptr + (feat_idx * d_model) + offsets
        row = tl.load(row_ptrs, mask=mask, other=0.0)

        acc += feat_val.to(tl.float32) * row.to(tl.float32)

    out_row = out_ptr + pid_token * d_model
    tl.store(out_row + offsets, acc, mask=mask)
