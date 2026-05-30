"""Tests for the CSR layout assumptions the kernel depends on.

These don't test the kernel's math — they test that the data layout fed to
the kernel is structured the way the kernel assumes. The load-bearing
assumption is that torch.nonzero returns coordinates sorted by token, so
that each token's features form a contiguous chunk in the flat arrays.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
DEVICE = "cuda"


def test_nonzero_is_token_sorted():
    """The CSR scheme breaks silently if nonzero() isn't sorted by token.
    Construct fires in deliberately scrambled order and confirm the
    returned token-id column is non-decreasing."""
    acts = torch.zeros(4, 10, device=DEVICE)
    # fire in scrambled (token, feature) order
    acts[2, 1] = 1.0
    acts[0, 7] = 1.0
    acts[3, 0] = 1.0
    acts[1, 5] = 1.0
    acts[0, 2] = 1.0

    coords = acts.nonzero()
    token_ids = coords[:, 0]
    # must be non-decreasing: all of token 0, then 1, then 2, ...
    assert torch.all(
        token_ids[1:] >= token_ids[:-1]
    ), f"nonzero not token-sorted: {token_ids.tolist()}"


def test_row_offsets_bracket_each_token():
    """row_offsets[b]:row_offsets[b+1] must select exactly token b's features."""
    B, n_features = 4, 20
    acts = torch.zeros(B, n_features, device=DEVICE)
    counts_expected = [3, 0, 5, 2]
    g = torch.Generator(device=DEVICE).manual_seed(0)
    for b, c in enumerate(counts_expected):
        if c:
            fired = torch.randperm(n_features, generator=g, device=DEVICE)[:c]
            acts[b, fired] = 1.0

    coords = acts.nonzero()
    token_ids = coords[:, 0]
    counts = torch.bincount(token_ids, minlength=B)
    row_offsets = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
    row_offsets[1:] = counts.cumsum(0).to(torch.int32)

    assert counts.tolist() == counts_expected
    # each bracketed slice should have all-matching token ids
    for b in range(B):
        s, e = row_offsets[b].item(), row_offsets[b + 1].item()
        chunk_tokens = token_ids[s:e]
        assert torch.all(
            chunk_tokens == b
        ), f"token {b}'s chunk [{s}:{e}] contains wrong tokens: {chunk_tokens.tolist()}"
