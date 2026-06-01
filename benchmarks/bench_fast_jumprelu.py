"""End-to-end benchmark: stock SAELens JumpReLU decode vs. our sparse kernel.

Loads a pretrained JumpReLU SAE, builds a decode() that swaps only the
decoder matmul for our sparse kernel (preserving bias, hooks, norm, and
reshape), then checks:
  1. correctness — the override must match stock decode (allclose)
  2. speed       — full decode wall-clock, stock vs. sparse

This measures the FULL decode, so the speedup is diluted vs. the raw-matmul
benchmark (and currently throttled by build_csr overhead). The point here is
contract correctness in context + the realistic dilution factor, not the
headline kernel speedup.

Run:  python benchmarks/bench_sae_decode.py
"""

import torch
import triton

from kernel_jumprelu_sae.exact.wrappers import sparse_decode

DEVICE = "cuda"

# A real JumpReLU release. Adjust if you have a different one downloaded.
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_65k/canonical"

# Benchmark config
B = 32  # batch (tokens)
L0 = 72  # active features per token (Gemma Scope avg L0 ~ 72)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load_sae():
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device=DEVICE)
    if isinstance(sae, tuple):  # some versions return (sae, cfg, sparsity)
        sae = sae[0]
    return sae


def make_feature_acts(B, n_features, L0, dtype, seed=0):
    """Sparse [B, n_features] activations, L0 active per token, positive values
    (post-JumpReLU activations are non-negative)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    acts = torch.zeros(B, n_features, device=DEVICE, dtype=dtype)
    for b in range(B):
        fired = torch.randperm(n_features, generator=g, device=DEVICE)[:L0]
        acts[b, fired] = torch.randn(L0, generator=g, device=DEVICE).abs().to(dtype)
    return acts


# ---------------------------------------------------------------------------
# our decode: swap ONLY the matmul, preserve every other step verbatim
# ---------------------------------------------------------------------------
def make_fast_decode(sae):
    """Return a decode() mirroring sae.decode but with the decoder matmul
    routed through the sparse kernel. Captures the SAE's own modules/params so
    bias, hooks, norm, and reshape behave identically."""

    def fast_decode(feature_acts):
        # original: feature_acts @ self.W_dec + self.b_dec
        sae_out_pre = sparse_decode(feature_acts, sae.W_dec) + sae.b_dec
        # everything below is copied verbatim from JumpReLUSAE.decode
        sae_out_pre = sae.hook_sae_recons(sae_out_pre)
        sae_out_pre = sae.run_time_activation_norm_fn_out(sae_out_pre)
        return sae.reshape_fn_out(sae_out_pre, sae.d_head)

    return fast_decode


def main():
    sae = load_sae()
    n_features, d_model = sae.W_dec.shape
    dtype = sae.W_dec.dtype
    print(f"release={SAE_RELEASE}  sae_id={SAE_ID}")
    print(f"n_features={n_features}  d_model={d_model}  dtype={dtype}  B={B}  L0={L0}")

    acts = make_feature_acts(B, n_features, L0, dtype=dtype)

    # quick shape sanity: report what decode actually receives
    print(
        f"feature_acts.shape = {tuple(acts.shape)}  (kernel expects 2D [B, n_features])"
    )

    stock_decode = sae.decode
    fast_decode = make_fast_decode(sae)

    # --- correctness: the override must match stock decode ---
    with torch.no_grad():
        out_stock = stock_decode(acts)
        out_fast = fast_decode(acts)

    if out_stock.shape != out_fast.shape:
        print(
            f"SHAPE MISMATCH: stock {tuple(out_stock.shape)} vs fast {tuple(out_fast.shape)}"
        )
        return

    max_diff = (out_stock.float() - out_fast.float()).abs().max().item()
    # looser tolerance for half precision
    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3
    ok = torch.allclose(out_stock.float(), out_fast.float(), atol=atol, rtol=1e-2)
    print(
        f"correctness: max_diff={max_diff:.2e}  tol={atol}  {'OK' if ok else 'MISMATCH'}"
    )
    if not ok:
        print(
            "  contract mismatch — likely dtype, a missing decode step, or shape handling"
        )

    # --- speed: full decode, stock vs. sparse ---
    with torch.no_grad():
        stock_ms = triton.testing.do_bench(lambda: stock_decode(acts))
        fast_ms = triton.testing.do_bench(lambda: fast_decode(acts))

    print(f"stock  decode: {stock_ms:.3f} ms")
    print(f"sparse decode: {fast_ms:.3f} ms")
    print(f"speedup:       {stock_ms / fast_ms:.2f}x")


if __name__ == "__main__":
    main()
