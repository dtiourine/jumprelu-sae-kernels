# JumpReLU SAE Kernels

Triton GPU kernels that accelerate the decode step of a [JumpReLU Sparse Autoencoder](https://arxiv.org/abs/2407.14435).

In a JumpReLU SAE, the reconstruction is computed as a dense matmul `feature_acts @ W_dec`, but `feature_acts` is highly sparse. Only a small number of features (`L0`) fire out of a large dictionary (`n_features`), so the dense matmul spends almost all of its FLOPs multiplying by zeros. These kernels skip the zeros: they compact each token's fired features into a CSR-style layout and accumulate only the decoder rows that actually contribute.


## Installation

This project uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # runtime install
uv sync --extra dev     # include dev tools (pytest, ruff, etc.)
```

A CUDA GPU is required. The Triton kernels cannot run on CPU.

## Usage

```python
import torch
from jumprelu_sae_kernels import sparse_decode

# feature_acts: [B, n_features] sparse activations
# W_dec:        [n_features, d_model] decoder weights
with torch.no_grad():
    out = sparse_decode(feature_acts, W_dec)   # == feature_acts @ W_dec
```

Currently, the kernel is inference-only (it is not autograd-aware). Call it under `torch.no_grad()` or `inference_mode`.

Two variants are available via `sparse_decode(..., variant=...)`:

- **`"exact"`** (default): true CSR layout. Correct for any `L0`, no wasted memory.
- **`"fixed"`**: fixed-stride layout. Faster, but over-allocates and requires `max_l0` to bound the true `L0`.

For the `"fixed"` variant, `validate` controls what happens when a token fires more features than `max_l0`:

- **`validate=True`** (default): checks whether any token exceeded `max_l0` and raises `ValueError` if so, guarding against silently truncated (incorrect) output. This costs a GPU→CPU sync per call.
- **`validate=False`**: skips the check for the sync-free fast path. Only safe when `max_l0` is known to bound `L0` — an over-`max_l0` token is silently truncated.

## Benchmarks

End-to-end decode swap on real SAEs, replacing the stock dense decode (`B = 32`, fp32, NVIDIA RTX 4090). Speedup is over PyTorch's dense `feature_acts @ W_dec`.

| SAE | width | d_model | stock (ms) | `exact` | `fixed` (validate) | `fixed` (no validate, unsafe) |
|-----|------:|--------:|-----------:|--------:|-------------------:|--------:|
| Gemma Scope 2B, layer 12 (65k) | 65,536 | 2,304 | 0.71 | 3.9× | 5.5× | 11.3× |
| Gemma Scope 2B, layer 20 (65k) | 65,536 | 2,304 | 0.71 | 4.3× | 5.6× | 11.4× |
| Gemma Scope 2B, layer 12 (262k) | 262,144 | 2,304 | 2.63 | 12.1× | 14.5× | 22.6× |
| Gemma Scope 9B, layer 20 (65k) | 65,536 | 3,584 | 1.06 | 5.7× | 7.3× | 13.3× |
| Qwen Scope 3.5-2B, layer 12 (32k) | 32,768 | 2,048 | 0.36 | 2.0× | 2.5× | 5.7× |

Columns are speedup over PyTorch's dense decode. `fixed (validate)` is `validate=True`; `fixed (no validate)` is `validate=False`.

The speedup grows with the dictionary size and the sparsity of the activations, since those are exactly the FLOPs the dense matmul wastes. Full sweeps over `L0`, batch size, `d_model`, and `n_features`, along with comparisons against `torch.compile` and `torch.sparse`, live in [`benchmarks/`](benchmarks/).

## Development

```bash
uv run pytest           # run the test suite (auto-skips without CUDA)
uv run ruff check .     # lint
```

## License

[MIT](LICENSE)
