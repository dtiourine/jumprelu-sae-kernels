import triton
import triton.language as tl


@triton.jit
def count_nonzero(
    feature_acts_ptr, counts_ptr, n_features, BLOCK_F: tl.constexpr
):
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
