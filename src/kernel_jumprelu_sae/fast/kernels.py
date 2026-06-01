import triton
import triton.language as tl


@triton.jit
def compute_csr_kernel_fast(
    feature_acts_ptr,
    write_pos_ptr,  # per-token cursor (atomics); AFTER kernel = per-token count
    flat_idx_ptr,
    flat_val_ptr,
    n_features,
    max_l0,
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

    region_start = pid_token * max_l0
    block_count = tl.sum(fired_int)
    base = tl.atomic_add(write_pos_ptr + pid_token, block_count)
    local_rank = tl.cumsum(fired_int) - fired_int

    local_slot = base + local_rank
    in_region = local_slot < max_l0  # overflow guard
    write_mask = fired & mask & in_region

    slots = region_start + local_slot
    tl.store(flat_idx_ptr + slots, feat_offsets.to(tl.int32), mask=write_mask)
    tl.store(flat_val_ptr + slots, vals, mask=write_mask)


@triton.jit
def sparse_decode_kernel_fast(
    flat_idx_ptr,
    flat_val_ptr,
    counts_ptr,  # per-token actual count
    W_dec_ptr,
    out_ptr,
    d_model,
    max_l0,  # fixed region stride
    BLOCK_D: tl.constexpr,
):
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)

    start = pid_token * max_l0
    n = tl.load(counts_ptr + pid_token)

    offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < d_model
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in range(n):
        j = start + i
        feat_idx = tl.load(flat_idx_ptr + j)
        feat_val = tl.load(flat_val_ptr + j)
        row_ptrs = W_dec_ptr + feat_idx * d_model + offsets
        row = tl.load(row_ptrs, mask=mask, other=0.0)
        acc += feat_val * row

    out_row = out_ptr + pid_token * d_model
    tl.store(out_row + offsets, acc, mask=mask)
