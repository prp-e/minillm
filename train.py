from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import torch, torch.nn.functional as F, math, random, numpy as np, os, time, pickle
import json 

def load_model_cfg(json_file):
    """Reading configurations from a json file"""
    with open(json_file) as f:
        cfg = json.load(f)
    return cfg

def set_seed(seed=42):
    """Fix random seeds for reproducibility"""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def repeat_kv(hidden_states, n_rep):
    """Repeat key/value heads for GQA"""
    b, n_kv, s, d = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden_states.reshape(b, n_kv * n_rep, s, d)

def zeropower_via_newtonschulz5(G, steps=5):
    """Approximate orthogonalization via Newton–Schulz iteration"""
    assert G.ndim >= 2
    a,b,c = 3.4445,-4.7750,2.0315
    X = G.bfloat16()
    if G.size(-2) > G.size(-1): X = X.mT
    X = X / (X.norm(dim=(-2,-1),keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b*A + c*A@A
        X = a*X + B@X
    if G.size(-2) > G.size(-1): X = X.mT
    return X

def muon_step(params, grads, states, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
    """One Muon optimizer step"""
    with torch.no_grad():
        for p,g in zip(params, grads):
            if g is None: continue
            buf = states.setdefault(p, torch.zeros_like(g))
            buf.lerp_(g, 1 - momentum)
            g = g.lerp(buf, momentum) if nesterov else buf
            g = zeropower_via_newtonschulz5(g, steps=ns_steps)
            p.add_(g, alpha=-lr * math.sqrt(max(1, p.size(-2)/p.size(-1))))

if __name__ == "__main__":
    print("training your model here.")