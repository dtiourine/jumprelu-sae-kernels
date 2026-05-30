from src.kernel import sparse_decode_kernel
import triton
import torch


def sparse_decode(idx, val, W_dec, BLOCK_D=256):
    d_model = W_dec.shape[1]
    L0 = idx.shape[0]

    out = torch.zeros(d_model, device=W_dec.device, dtype=torch.float32)

    grid = (triton.cdiv(d_model, BLOCK_D),)
    
    sparse_decode_kernel[grid](
        idx, val, W_dec, out, L0, d_model, BLOCK_D=BLOCK_D
    )
    
    return out 
