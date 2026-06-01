"""Layer 4 — layout invariants the decode kernels depend on. These don't test
math; they pin the data layout. Covers both variants:
  - exact: torch.nonzero / cumsum ordering makes each token's features one
    contiguous chunk bracketed by row_offsets.
  - fixed: each token's region is exactly [b*max_l0, b*max_l0+count); overflow
    is dropped, not spilled into the neighbor's region.
"""

import pytest

from conftest import requires_cuda, make_sparse_acts
from jumprelu_sae_kernels.exact.wrappers import build_csr as build_csr_exact
from jumprelu_sae_kernels.fixed.wrappers import build_csr as build_csr_fixed

pytestmark = requires_cuda


# --- exact variant -------------------------------------------------------
def test_exact_row_offsets_bracket_each_token():
    B, n_features = 4, 1024
    counts_expected = [3, 0, 5, 2]
    acts = make_sparse_acts(B, n_features, counts_expected)
    flat_idx, flat_val, row_offsets, B_ = build_csr_exact(acts)
    assert B_ == B
    # row_offsets are monotonic and the gaps equal the per-token counts
    gaps = (row_offsets[1:] - row_offsets[:-1]).tolist()
    assert gaps == counts_expected
    # each token's bracketed indices are real fired features of that token
    for b in range(B):
        s, e = int(row_offsets[b]), int(row_offsets[b + 1])
        idxs = flat_idx[s:e].tolist()
        nz = acts[b].nonzero().squeeze(-1).tolist()
        assert sorted(idxs) == sorted(nz)


# --- fixed variant -------------------------------------------------------
def test_fixed_region_is_exact_stride():
    B, n_features, max_l0 = 4, 1024, 16
    counts_expected = [3, 0, 5, 2]
    acts = make_sparse_acts(B, n_features, counts_expected)
    flat_idx, flat_val, counts, B_, ml0 = build_csr_fixed(acts, max_l0=max_l0)
    assert B_ == B and ml0 == max_l0
    assert counts.tolist() == counts_expected
    for b in range(B):
        s = b * max_l0
        c = counts_expected[b]
        idxs = flat_idx[s:s + c].tolist()
        nz = acts[b].nonzero().squeeze(-1).tolist()
        assert sorted(idxs) == sorted(nz)


def test_fixed_overflow_dropped_not_spilled():
    """A token firing > max_l0 makes build_csr raise rather than silently
    truncating (the in-kernel write guard prevents spill into the neighbor's
    region; the wrapper then refuses the result). The no-corruption guarantee
    for in-bounds batches is covered by test_fixed_overflow_does_not_corrupt_neighbor
    in test_edge_cases.py."""
    B, n_features, max_l0 = 2, 1024, 4
    acts = make_sparse_acts(B, n_features, l0_per_token=[10, 2])
    with pytest.raises(ValueError, match="max_l0"):
        build_csr_fixed(acts, max_l0=max_l0)
