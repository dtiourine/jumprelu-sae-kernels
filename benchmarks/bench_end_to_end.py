"""End-to-end: stock SAELens JumpReLU decode vs our sparse kernel swapped in for
only the decoder matmul. Checks contract correctness in context (under no_grad)
and reports the realistic, diluted full-decode speedup.

Run:  uv run python -m benchmarks.bench_end_to_end
Skips gracefully if the SAE weights aren't available.
Writes benchmarks/results/end_to_end.{csv,json} on success.
"""

import torch

from benchmarks.lib.harness import capture_env, bench, write_results
from jumprelu_sae_kernels import sparse_decode

SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_20/width_65k/canonical"
B, L0 = 32, 72  # Gemma Scope avg L0 ~ 72


def load_sae():
    from sae_lens import SAE
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=SAE_ID, device="cuda")
    return sae[0] if isinstance(sae, tuple) else sae


def make_feature_acts(B, n_features, L0, dtype, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    acts = torch.zeros(B, n_features, device="cuda", dtype=dtype)
    for b in range(B):
        fired = torch.randperm(n_features, generator=g, device="cuda")[:L0]
        acts[b, fired] = torch.randn(L0, generator=g, device="cuda").abs().to(dtype)
    return acts


def make_fast_decode(sae, variant, max_l0):
    # No detach needed: the benchmark runs under torch.no_grad(), and
    # sparse_decode's inference-only guard is grad-context-aware — it allows
    # no_grad inference even though sae.W_dec carries requires_grad=True.
    def fast_decode(feature_acts):
        sae_out_pre = sparse_decode(
            feature_acts, sae.W_dec, variant=variant, max_l0=max_l0
        ) + sae.b_dec
        sae_out_pre = sae.hook_sae_recons(sae_out_pre)
        sae_out_pre = sae.run_time_activation_norm_fn_out(sae_out_pre)
        return sae.reshape_fn_out(sae_out_pre, sae.d_head)
    return fast_decode


def main():
    try:
        sae = load_sae()
    except Exception as e:
        print(f"SKIP: could not load SAE ({type(e).__name__}: {e})")
        return

    n_features, d_model = sae.W_dec.shape
    dtype = sae.W_dec.dtype
    acts = make_feature_acts(B, n_features, L0, dtype=dtype)
    # fixed variant needs max_l0 >= the actual max L0 in the batch
    max_l0 = int((acts != 0).sum(dim=1).max().item())
    stock = sae.decode
    atol = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3

    rows = []
    with torch.no_grad():
        stock_ms = bench(lambda: stock(acts))["median_ms"]
        out_stock = stock(acts)
        print(f"stock decode: {stock_ms:.3f}ms")
        for variant in ("exact", "fixed"):
            fast = make_fast_decode(sae, variant, max_l0)
            out_fast = fast(acts)
            max_diff = (out_stock.float() - out_fast.float()).abs().max().item()
            ok = torch.allclose(out_stock.float(), out_fast.float(), atol=atol, rtol=1e-2)
            fast_ms = bench(lambda: fast(acts))["median_ms"]
            speedup = stock_ms / fast_ms
            print(f"  {variant:5s}: max_diff={max_diff:.2e} tol={atol} "
                  f"{'OK' if ok else 'MISMATCH'}  sparse={fast_ms:.3f}ms "
                  f"speedup={speedup:.2f}x")
            if ok:
                rows.append(dict(
                    release=SAE_RELEASE, sae_id=SAE_ID, variant=variant,
                    n_features=int(n_features), d_model=int(d_model),
                    dtype=str(dtype), B=B, L0=L0, max_l0=max_l0,
                    max_diff=max_diff, stock_ms=stock_ms, fast_ms=fast_ms,
                    speedup=speedup,
                ))

    if rows:
        write_results(rows, "end_to_end", capture_env())


if __name__ == "__main__":
    main()
