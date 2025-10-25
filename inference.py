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