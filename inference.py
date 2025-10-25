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