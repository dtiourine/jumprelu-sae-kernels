import torch
import triton
import triton.language as tl


@triton.jit
def extract_kernel(
    z_ptr,
    threshold_ptr,
    idx_out_ptr,
    val_out_ptr,
    counter_ptr,
    n_features,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_features

    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)
    thresh = tl.load(threshold_ptr + offsets, mask=mask, other=0.0)
    fired = (z > thresh) & mask


@triton.jit
def sparse_decode_kernel(
    idx_ptr, val_ptr, W_dec_ptr, out_ptr, L0, d_model, BLOCK_D: tl.constexpr
):
    """Sparse SAE decoder for a single token.

    Computes the reconstruction as a weighted sum of the decoder rows
    belonging to the features that fired:

        x_hat = sum_i  val[i] * W_dec[idx[i], :]

    Only the L0 fired features are touched, so the cost scales with L0
    rather than the full feature count. Each program instance owns a
    BLOCK_D-wide slice of the (dense) output and loops over the fired
    features, accumulating each one's contribution into that slice.

    Args:
        idx_ptr: Pointer to an int array of shape [L0] holding the indices
            of the features that fired (i.e. which rows of W_dec to read).
        val_ptr: Pointer to a float array of shape [L0] holding the
            activation value of each fired feature, aligned with idx_ptr.
        W_dec_ptr: Pointer to the decoder weight matrix, shape
            [n_features, d_model], stored row-major.
        out_ptr: Pointer to the output buffer of shape [d_model] where the
            reconstruction x_hat is written.
        L0: Number of features that fired for this token; the loop length.
        d_model: Output dimension, equal to the row width of W_dec.
        BLOCK_D: Compile-time tile width — how many output columns each
            program instance is responsible for.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offsets < d_model

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for i in range(L0):
        feat_idx = tl.load(idx_ptr + i)
        feat_val = tl.load(val_ptr + i)

        row_ptrs = W_dec_ptr + feat_idx * d_model + offsets
        row = tl.load(row_ptrs, mask=mask, other=0.0)

        acc += feat_val * row

    tl.store(out_ptr + offsets, acc, mask=mask)
