import torch

DEVICE = "cuda"


def fixed_l0_feature_acts(B, n_features, L0, dtype=torch.float32, seed=0):
    """Generate dummy feature_acts where every token fires exactly L0 features."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    acts = torch.zeros(B, n_features, device=DEVICE, dtype=dtype)
    for b in range(B):
        fired = torch.randperm(n_features, generator=g, device=DEVICE)[:L0]
        acts[b, fired] = torch.randn(L0, generator=g, device=DEVICE).abs().to(dtype)
    return acts


def ragged_l0_feature_acts(B, n_features, mean_l0, dtype=torch.float32, seed=0, spread=0.4):
    """Generate dummy feature_acts with per-token L0 varying around mean_l0. This is
    closer to a real SAE's firing distribution than a fixed count.

    Returns (acts, max_l0); max_l0 is the bound the fixed variant needs to stay
    correct (a too-small bound truncates fired features).
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    sigma = max(1.0, mean_l0 * spread)
    l0 = torch.normal(
        mean=float(mean_l0), std=sigma, size=(B,), generator=g, device=DEVICE
    )
    l0 = l0.round().clamp(0, n_features).to(torch.int64)
    acts = torch.zeros(B, n_features, device=DEVICE, dtype=dtype)
    for b in range(B):
        k = int(l0[b])
        if k > 0:
            fired = torch.randperm(n_features, generator=g, device=DEVICE)[:k]
            acts[b, fired] = torch.randn(k, generator=g, device=DEVICE).abs().to(dtype)
    return acts, int(l0.max().item())
