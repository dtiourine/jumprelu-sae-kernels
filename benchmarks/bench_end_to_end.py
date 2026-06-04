import torch

from benchmarks.lib.harness import capture_env, bench, write_results
from jumprelu_sae_kernels import sparse_decode

B = 32

SAES = [
    # Gemma Scope 2B, layer 20, width 65k (d_model 2304).
    dict(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_20/width_65k/canonical",
        L0=72,
    ),
    # Gemma Scope 9B (d_model 3584).
    dict(
        release="gemma-scope-9b-pt-res-canonical",
        sae_id="layer_20/width_65k/canonical",
        L0=72,
    ),
    # Different layer, same model
    dict(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_65k/canonical",
        L0=72,
    ),
    dict(
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_262k/canonical",
        L0=100,
    ),
    # Qwen-scope (Scope-style JumpReLU SAE, target L0 = 100)
    dict(release="qwen-scope-3.5-2b-base-w32k-l100", sae_id="layer12", L0=100),
]


def load_sae(release, sae_id):
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device="cuda")
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
        sae_out_pre = (
            sparse_decode(feature_acts, sae.W_dec, variant=variant, max_l0=max_l0)
            + sae.b_dec
        )
        sae_out_pre = sae.hook_sae_recons(sae_out_pre)
        sae_out_pre = sae.run_time_activation_norm_fn_out(sae_out_pre)
        return sae.reshape_fn_out(sae_out_pre, sae.d_head)

    return fast_decode


def bench_sae(cfg):
    release, sae_id, L0 = cfg["release"], cfg["sae_id"], cfg["L0"]
    label = f"{release} :: {sae_id}"
    try:
        sae = load_sae(release, sae_id)
    except Exception as e:
        print(f"SKIP {label} ({type(e).__name__}: {e})")
        return []

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
        print(f"{label}")
        print(
            f"  n_features={n_features} d_model={d_model} dtype={dtype} "
            f"L0={L0} stock={stock_ms:.3f}ms"
        )
        for variant in ("exact", "fixed"):
            fast = make_fast_decode(sae, variant, max_l0)
            out_fast = fast(acts)
            max_diff = (out_stock.float() - out_fast.float()).abs().max().item()
            ok = torch.allclose(
                out_stock.float(), out_fast.float(), atol=atol, rtol=1e-2
            )
            fast_ms = bench(lambda: fast(acts))["median_ms"]
            speedup = stock_ms / fast_ms
            print(
                f"  {variant:5s}: max_diff={max_diff:.2e} tol={atol} "
                f"{'OK' if ok else 'MISMATCH'}  sparse={fast_ms:.3f}ms "
                f"speedup={speedup:.2f}x"
            )
            if ok:
                rows.append(
                    dict(
                        release=release,
                        sae_id=sae_id,
                        variant=variant,
                        n_features=int(n_features),
                        d_model=int(d_model),
                        dtype=str(dtype),
                        B=B,
                        L0=L0,
                        max_l0=max_l0,
                        max_diff=max_diff,
                        stock_ms=stock_ms,
                        fast_ms=fast_ms,
                        speedup=speedup,
                    )
                )
    return rows


def main():
    rows = []
    for cfg in SAES:
        rows.extend(bench_sae(cfg))

    if rows:
        write_results(rows, "end_to_end", capture_env())
        print(
            f"\nwrote {len(rows)} rows across {len({r['release'] + r['sae_id'] for r in rows})} SAEs"
        )
    else:
        print("\nno SAEs benchmarked (all skipped) — nothing written")


if __name__ == "__main__":
    main()
