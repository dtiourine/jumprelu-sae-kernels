import triton
import triton.language as tl


@triton.jit
def count_nonsparse_elements(feature_acts_ptr, counts_ptr, n_features, BLOCK_F: tl.constexpr):
    pid_token = tl.program_id(0)
    pid_d = tl.program_id(1)
    feat_offsets = pid_d * BLOCK_F + tl.arange(0, BLOCK_F)
    mask = feat_offsets < n_features

    vals = tl.load(
        feature_acts_ptr + pid_token * n_features + feat_offsets, mask=mask, other=0.0
    )
    fired = vals != 0.0
    block_fired_count = tl.sum(fired.to(tl.int32))
    tl.atomic_add(counts_ptr + pid_token, block_fired_count)
