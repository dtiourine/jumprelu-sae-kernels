from jumprelu_sae_kernels.exact.wrappers import sparse_decode as _decode_exact
from jumprelu_sae_kernels.fixed.wrappers import sparse_decode as _decode_fixed


def sparse_decode(feature_acts, W_dec, variant="exact", max_l0=512):
    """Sparse replacement for feature_acts @ W_dec.

    variant: "exact" — any L0, no wasted memory (default, always correct)
             "fixed" — faster, requires L0 <= max_l0, over-allocates

    Inference-only: this kernel is not autograd-aware. Passing inputs that
    require grad raises rather than silently returning a detached result.
    """
    if feature_acts.requires_grad or W_dec.requires_grad:
        raise RuntimeError(
            "sparse_decode is inference-only and not autograd-aware; it does "
            "not support backprop. Call it under torch.no_grad() or on detached "
            "tensors. (Got requires_grad=True on an input.)"
        )
    if variant == "exact":
        return _decode_exact(feature_acts, W_dec)
    elif variant == "fixed":
        return _decode_fixed(feature_acts, W_dec, max_l0)
    raise ValueError(f"unknown variant: {variant!r}")
