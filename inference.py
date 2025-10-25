from transformers import AutoTokenizer
import math, torch
import torch.nn.functional as F
import random, numpy as np
import argparse

def set_seed(seed=42):
    """Fix random seeds to keep inference deterministic where possible."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads (view-based) to match query heads in GQA."""
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1: return hidden_states
    hs = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hs.reshape(b, n_kv * n_rep, s, d)

def rotary_emb(dim, max_seq_len):
    """Precompute RoPE cos/sin tables for dim and max_seq_len."""
    ang = (1/10000)**torch.linspace(0,1,steps=dim//4, dtype=torch.float32)
    ang = torch.cat([ang, torch.zeros_like(ang)])
    t = torch.arange(max_seq_len, dtype=torch.float32)
    theta = torch.einsum("i,j->ij", t, ang)
    return theta.cos(), theta.sin()

def apply_rotary(x, cos, sin):
    """Apply RoPE rotation to tensor shaped (B,T,H,D)."""
    x1, x2 = x.chunk(2, dim=-1)
    T = x.size(1)
    cos, sin = cos[:T], sin[:T]
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat((y1, y2), dim=-1)

def qwen_attention(x, attn, cos, sin, dropout=0.0):
    """Qwen-style attention with QK-norm, RoPE, GQA, Flash-attn."""
    b,t,d = x.shape
    n_heads, n_kv, d_k, d_model = attn["n_heads"], attn["n_kv"], attn["d_k"], attn["d_model"]
    q = F.linear(x, attn["wq"])  
    k = F.linear(x, attn["wk"])  
    v = F.linear(x, attn["wv"])  
    q = q.view(b,t,n_heads,d_k); k = k.view(b,t,n_kv,d_k); v = v.view(b,t,n_kv,d_k)
    q = F.normalize(q, dim=-1);  k = F.normalize(k, dim=-1)
    q = apply_rotary(q, cos, sin); k = apply_rotary(k, cos, sin)
    Q, K, V = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
    K = repeat_kv(K, attn["n_groups"]); V = repeat_kv(V, attn["n_groups"])
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=dropout)
    out = out.transpose(1,2).contiguous().view(b,t,d_model)
    return F.linear(out, attn["wo"])

def swiglu_ffn(x, w_up, w_gate, w_down, dropout=0.0):
    """Feed-forward with SwiGLU activation."""
    a = F.silu(F.linear(x, w_gate)) * F.linear(x, w_up)
    return F.linear(F.dropout(a, dropout, training=False), w_down)