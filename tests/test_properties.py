"""Layer 3 — property-based fuzzing. The dense matmul is the oracle; hypothesis
explores a bounded input space and shrinks any failure to a minimal repro.
Bounds are kept modest so generated tensors fit comfortably on a 24GB 4090."""

import torch
from hypothesis import given, settings, strategies as st

from conftest import (
    DEVICE,
    requires_cuda,
    make_sparse_acts,
    dense_fp32_ref,
    decode_variant,
)

pytestmark = requires_cuda

_dtypes = st.sampled_from([torch.float32, torch.float16, torch.bfloat16])
_variants = st.sampled_from(["exact", "fixed"])


@settings(deadline=None, max_examples=150)
@given(
    B=st.integers(min_value=1, max_value=16),
    n_features=st.integers(min_value=4, max_value=4096),
    d_model=st.integers(min_value=1, max_value=1024),
    dtype=_dtypes,
    variant=_variants,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    data=st.data(),
)
def test_property_matches_dense(B, n_features, d_model, dtype, variant, seed, data):
    # ragged per-token L0 in [0, n_features]
    l0 = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=n_features),
            min_size=B, max_size=B,
        )
    )
    W_dec = torch.randn(n_features, d_model, device=DEVICE, dtype=dtype)
    acts = make_sparse_acts(B, n_features, l0, dtype=dtype, seed=seed)
    max_l0 = max(max(l0), 1)
    out = decode_variant(acts, W_dec, variant, max_l0=max_l0)
    ref = dense_fp32_ref(acts, W_dec)
    assert out.dtype == torch.float32
    # The fuzzer draws L0 up to n_features, so it stresses fp32 accumulation
    # order over very long sums — a looser tolerance than the bounded-L0 (<=100)
    # correctness suite. The kernel is dtype-independent in accuracy (fp32
    # arithmetic), so one tolerance covers all dtypes here.
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=2e-3)
