from jumprelu_sae_kernels.exact.wrappers import sparse_decode as _decode_exact
from jumprelu_sae_kernels.fixed.wrappers import sparse_decode as _decode_fixed


def sparse_decode(feature_acts, W_dec, variant="exact", max_l0=512):
    """Sparse replacement for feature_acts @ W_dec.

    variant: "exact" — any L0, no wasted memory (default, always correct)
             "fixed" — faster, requires L0 <= max_l0, over-allocates
    """
    if variant == "exact":
        return _decode_exact(feature_acts, W_dec)
    elif variant == "fixed":
        return _decode_fixed(feature_acts, W_dec, max_l0)
    raise ValueError(f"unknown variant: {variant!r}")
