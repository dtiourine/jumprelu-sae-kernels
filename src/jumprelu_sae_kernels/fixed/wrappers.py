import torch
import triton

from jumprelu_sae_kernels.fixed.kernels import (
    compute_csr_kernel,
    sparse_decode_kernel,
)


def build_csr(
    feature_acts: torch.Tensor,
    BLOCK_F: int = 1024,
    max_l0: int = 512,
    validate: bool = True,
):
    """Build a fixed-stride CSR layout from dense [B, n_features] activations.

    Each token gets a fixed region of `max_l0` slots at [b*max_l0, (b+1)*max_l0),
    so region starts are pure arithmetic (no count pass, no cumsum, no CPU sync).
    write_pos doubles as the per-token write cursor during the kernel and the
    per-token count afterward. The kernel always guards against writing past a
    token's region (no memory corruption regardless of `validate`).

    `validate` controls the overflow *check*, not the memory safety:
      * True (default) — verify no token fired more than max_l0 features, raising
        ValueError if one did. This costs a `.item()` GPU->CPU sync per call.
      * False — skip the check (no sync) for the fast path. A token exceeding
        max_l0 is then SILENTLY TRUNCATED, producing an incorrect result with no
        error. Only use this when max_l0 is known to bound L0.
    """
    B, n_features = feature_acts.shape
    device = feature_acts.device

    capacity = B * max_l0
    flat_idx = torch.empty(capacity, dtype=torch.int32, device=device)
    flat_val = torch.empty(capacity, dtype=feature_acts.dtype, device=device)

    write_pos = torch.zeros(B, dtype=torch.int32, device=device)

    grid = (B, triton.cdiv(n_features, BLOCK_F))
    compute_csr_kernel[grid](
        feature_acts,
        write_pos,
        flat_idx,
        flat_val,
        n_features,
        max_l0,
        BLOCK_F=BLOCK_F,
    )

    counts = write_pos  # final cursor = per-token count (stays on GPU)

    if validate:
        observed = counts.max().item()  # forces a GPU->CPU sync
        if observed > max_l0:
            raise ValueError(
                f"A token fired more than max_l0={max_l0} features "
                f"(max was {observed}). Increase max_l0 to at least this value "
                f"for exact results, or pass validate=False to skip this check "
                f"(which silently truncates instead of raising)."
            )

    return flat_idx, flat_val, counts, B, max_l0


def _sparse_decode(flat_idx, flat_val, counts, W_dec, B, max_l0, BLOCK_D: int = 256):
    """Launch the batched CSR decoder. Internal: handles grid + output allocation."""
    d_model = W_dec.shape[1]
    out = torch.zeros((B, d_model), device=W_dec.device, dtype=torch.float32)

    grid = (B, triton.cdiv(d_model, BLOCK_D))
    sparse_decode_kernel[grid](
        flat_idx, flat_val, counts, W_dec, out, d_model, max_l0, BLOCK_D=BLOCK_D
    )
    return out


def sparse_decode(feature_acts, W_dec, max_l0: int = 512, validate: bool = True):
    """Sparse replacement for `feature_acts @ W_dec`.

    Args:
        feature_acts: [B, n_features] sparse activations.
        W_dec: [n_features, d_model] decoder weights.
        max_l0: fixed per-token capacity; must be >= the max L0 across tokens.
        validate: if True (default), raise ValueError when a token fires more
            than max_l0 features (costs a GPU->CPU sync). If False, skip the
            check for the sync-free fast path — an over-max_l0 token is then
            silently truncated. Only pass False when max_l0 is known to bound L0.
    Returns:
        [B, d_model] reconstruction, equal to feature_acts @ W_dec.
    """
    W_dec = W_dec.contiguous()
    flat_idx, flat_val, counts, B, ml0 = build_csr(
        feature_acts, max_l0=max_l0, validate=validate
    )
    return _sparse_decode(flat_idx, flat_val, counts, W_dec, B, ml0)
