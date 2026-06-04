import torch

from jumprelu_sae_kernels.exact.wrappers import sparse_decode as _decode_exact
from jumprelu_sae_kernels.fixed.wrappers import sparse_decode as _decode_fixed


def sparse_decode(feature_acts, W_dec, variant="exact", max_l0=512, validate=True):
    """Sparse replacement for feature_acts @ W_dec.

    variant: "exact" — any L0, no wasted memory (default, always correct)
             "fixed" — faster, requires L0 <= max_l0, over-allocates

    validate: only affects variant="fixed". True (default) raises if a token
        fires more than max_l0 features, at the cost of a GPU->CPU sync per call.
        False skips the check for the sync-free fast path, silently truncating an
        over-max_l0 token — pass it only when max_l0 is known to bound L0. The
        "exact" variant ignores this (it has no max_l0 capacity).

    Inference-only: this kernel is not autograd-aware. If called in a
    grad-tracking context (autograd enabled) with an input that requires grad,
    it raises rather than silently returning a detached result that would break
    backprop. Under torch.no_grad()/inference_mode it runs normally even when
    parameters carry requires_grad=True (the common inference case), since no
    gradients are expected there.
    """
    if torch.is_grad_enabled() and (feature_acts.requires_grad or W_dec.requires_grad):
        raise RuntimeError(
            "sparse_decode is inference-only and not autograd-aware; it does "
            "not support backprop. Call it under torch.no_grad() or on detached "
            "tensors. (Got requires_grad=True on an input with autograd enabled.)"
        )
    if variant == "exact":
        return _decode_exact(feature_acts, W_dec)
    elif variant == "fixed":
        return _decode_fixed(feature_acts, W_dec, max_l0, validate=validate)
    raise ValueError(f"unknown variant: {variant!r}")
