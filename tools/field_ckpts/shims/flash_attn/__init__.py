"""CPU/MPS shim for flash_attn.flash_attn_func: identical math via torch SDPA (softmax scale 1/sqrt(head_dim))."""
import torch, torch.nn.functional as F
def flash_attn_func(q, k, v, causal=False, **kw):
    # q, k, v: (B, S, H, D) -> SDPA wants (B, H, S, D)
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    return y.transpose(1, 2).contiguous()
